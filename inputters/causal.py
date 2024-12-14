# coding=utf-8

import json
from math import ceil
from typing import List

import numpy as np
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from .PARAMS import GOLDEN_TRUTH, COMET_FEATURES, COMET_RELATIONS, SEK_RELATIONS, SUP_RELATIONS, CAUSAL_DF, EMOTIONS, STRATEGIES
from .inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
            self,
            dialog_id, emotion_id, input_ids, causal_attention_mask, causal_utt_attention_mask, strategy_history_input_ids,
            decoder_input_ids, labels, strategy_id
    ):
        self.dialog_id = dialog_id
        self.emotion_id = emotion_id
        self.causal_utt_attention_mask = causal_utt_attention_mask
        self.strategy_history_input_ids = strategy_history_input_ids
        self.strategy_history_input_length = len(strategy_history_input_ids)
        self.input_ids = input_ids
        self.causal_attention_mask = causal_attention_mask
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        self.strategy_id = strategy_id

        self.input_len = self.input_length + self.decoder_input_length

        # add comet features only for causal utterances
        self.comet_features = []
        for feature in COMET_FEATURES:
            utterance_feature = np.array([feature[dialog_id][i] for i in range(len(causal_utt_attention_mask))])
            self.comet_features.append(utterance_feature)


def featurize(
        bos, eos,
        emotion_id, dialog_id, 
        causal_attention_mask, causal_utt_attention_mask, 
        strategy_history, context, max_input_length,
        strategy_id, response, max_decoder_input_length
):
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]  # remove last eos
    
    causal_attention_mask = [c + [c[-1]] for c in causal_attention_mask]
    causal_attention_mask = sum(causal_attention_mask, [])[:-1]  # remove last eos
    strategy_history_input_ids = [strategy_history[0]] + [s + [eos] for s in strategy_history[1:]]
    strategy_history_input_ids = sum(strategy_history_input_ids, [])
    if len(strategy_history_input_ids) > 1:
        strategy_history_input_ids = strategy_history_input_ids[:-1]  # remove last eos

    situation_length = len(context[0])
    input_ids = input_ids[:situation_length] + input_ids[-max_input_length+situation_length:]
    causal_attention_mask = (causal_attention_mask[:situation_length] +
                             causal_attention_mask[-max_input_length+situation_length:])
    strategy_history_input_ids = strategy_history_input_ids[-max_input_length:]

    labels = (response + [eos])[:max_decoder_input_length]
    decoder_input_ids = [bos] + labels[:-1]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]
    assert len(input_ids) == len(causal_attention_mask)

    return InputFeatures(
        dialog_id,
        emotion_id,
        input_ids,
        causal_attention_mask,
        causal_utt_attention_mask,
        strategy_history_input_ids,
        decoder_input_ids,
        labels,
        strategy_id
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):

    def process(seq):
        return toker.convert_tokens_to_ids(toker.tokenize(seq))

    inputs = []
    dialog_id = data['dialog_id']
    emotion_id = EMOTIONS.index(data['emotion_type'])
    dialog = data['dialog']

    # situation_id = process('[Situation]')[0]
    # user_id = process('[User]')[0]

    situation = _norm(data['situation'])
    situation_input_ids = process(situation)
    
    context_input_ids = [situation_input_ids]
    causal_attention_mask = [[1]*len(situation_input_ids)]  # situation is cause
    causal_utt_attention_mask = [-1]
    strategy_history = [[toker.bos_token_id]]

    for i in range(len(dialog)):
        text: str = _norm(dialog[i]['text'])
        text_ids: List[int] = process(text)
        if dialog[i]['speaker'] == 'sys':
            strategy_id: int = process('[' + dialog[i]['strategy'] + ']')[0]
            if i > 0:
                res = {
                    'emotion_id': emotion_id,
                    'dialog_id': dialog_id,
                    'causal_attention_mask': causal_attention_mask.copy(),
                    'causal_utt_attention_mask': causal_utt_attention_mask.copy(),
                    'strategy_history': strategy_history.copy(),
                    'context': context_input_ids.copy(),
                    'response': text_ids.copy(),
                    'strategy_id': strategy_id,
                }
                inputs.append(res)

        # if dialog[i]['speaker'] == 'sys':
        #     text_ids = [strategy_id] + text_ids
        # else:
        #     text_ids = [user_id] + text_ids

        context_input_ids = context_input_ids + [text_ids]

        is_cause = CAUSAL_DF.loc[f"esconv.dialog_id_{dialog_id}_utt_{i + 1}"].preds
        causal_attention_mask = causal_attention_mask + [[is_cause] * len(text_ids)]
        causal_utt_attention_mask = causal_utt_attention_mask + [1 if dialog[i]['speaker'] == 'sys' else -1]
        if dialog[i]['speaker'] == 'sys':
            strategy_text = _norm(STRATEGIES[dialog[i]['strategy']])
            strategy_text_ids = process(strategy_text)
            strategy_history = strategy_history + [strategy_text_ids]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['emotion_id'], ipt['dialog_id'], ipt['causal_attention_mask'], ipt['causal_utt_attention_mask'], ipt['strategy_history'],
            ipt['context'],
            max_input_length, ipt['strategy_id'], ipt['response'], max_decoder_input_length
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        batch_size = len(features)
        comet_features = []
        # all 9 relations in COMET
        for i in range(len(COMET_RELATIONS)):
            comet_features.append(pad_sequence([torch.tensor(f.comet_features[i], dtype=torch.float) for f in features],
                                               batch_first=True, padding_value=pad))
        emotion_id = torch.tensor([f.emotion_id for f in features], dtype=torch.long)
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1] * f.input_length, dtype=torch.long) for f in features],
                                      batch_first=True, padding_value=0)
        strategy_history_input_ids = pad_sequence([torch.tensor(f.strategy_history_input_ids, dtype=torch.long) for f in features],
                                    batch_first=True, padding_value=pad)
        strategy_history_attention_mask = pad_sequence(
            [torch.tensor([1] * f.strategy_history_input_length, dtype=torch.long) for f in features],
            batch_first=True, padding_value=0)
        causal_attention_mask = pad_sequence(
            [torch.tensor(f.causal_attention_mask, dtype=torch.long) for f in features],
            batch_first=True, padding_value=0)
        causal_utt_attention_mask = pad_sequence([torch.tensor(f.causal_utt_attention_mask, dtype=torch.long) for f in features],
                                                 batch_first=True, padding_value=0)
        self_effect_mask = (causal_utt_attention_mask == -1).long()
        other_effect_mask = (causal_utt_attention_mask == 1).long()
        effect_context_attention_mask = torch.stack(
            len(SEK_RELATIONS) * [self_effect_mask] + len(SUP_RELATIONS) * [other_effect_mask],
            dim=2
        )
        effect_context_attention_mask = effect_context_attention_mask.flatten(start_dim=1, end_dim=2)
        encoder_attention_mask = torch.cat([
            attention_mask,
            effect_context_attention_mask
        ], dim=1)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            # decoder_attention_mask = pad_sequence(
            #     [torch.tensor([1] * f.decoder_input_length, dtype=torch.long) for f in features],
            #     batch_first=True,
            #     padding_value=0
            # )
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            # decoder_attention_mask = pad_sequence(
            #     [torch.tensor([1], dtype=torch.long) for f in features],
            #     batch_first=True,
            #     padding_value=0
            # )
            labels = None

        strategy_id = torch.tensor([f.strategy_id for f in features], dtype=torch.long) - len(toker) + 8

        res = {
            'emotion_id': emotion_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'causal_attention_mask': causal_attention_mask,
            'encoder_attention_mask': encoder_attention_mask,
            'comet_features': comet_features,

            'strategy_history_input_ids': strategy_history_input_ids,
            'strategy_history_attention_mask': strategy_history_attention_mask,

            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strategy_id': strategy_id
        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker):
    res = FeatureDataset.collate(features, toker, True)

    # res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strategy_id': 'pred_strategy_id',
    }

    if GOLDEN_TRUTH:
        other_res['cls_strategy_id'] = res.get('strategy_id')
    else:
        other_res['cls_strategy_id'] = res.pop('strategy_id')

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')
    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
