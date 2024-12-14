# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from .inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import (GOLDEN_TRUTH, COMET_FEATURES,
                     COMET_RELATIONS, PERSONA_RELATIONS, SEK_RELATIONS, STRATEGIES, SUP_RELATIONS,
                     CAUSAL_DF)


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
        dialog_id, situation_ids, speakers, input_ids,
        causal_attention_mask, strategy_history_input_ids,
        decoder_input_ids, labels, strategy_id
    ):
        self.situation_ids = situation_ids
        self.situation_length = len(situation_ids)
        
        self.speakers = speakers
        
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        
        self.causal_attention_mask = causal_attention_mask
        self.strategy_history_input_ids = strategy_history_input_ids
        self.strategy_history_input_length = len(strategy_history_input_ids)
        
        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        self.strategy_id = strategy_id

        self.input_len = self.input_length + self.decoder_input_length

        # add comet features
        self.situation_features = []
        self.utterance_features = []
        for feature in COMET_FEATURES:
            left, right = 1, 1+len(speakers)
            self.situation_features.append(np.array(feature[dialog_id][0]))
            self.utterance_features.append(np.array(feature[dialog_id][left:right]))

def featurize(
    bos, eos,
    dialog_id, situation_ids, speakers, context, max_input_length,
    causal_attention_mask, strategy_history,
    response, max_decoder_input_length, strategy_id
):
    context = [c + [eos] for c in context]
    input_ids: List = sum(context, [])[:-1]  # remove last eos
    causal_attention_mask = [c + [c[-1]] for c in causal_attention_mask]
    causal_attention_mask = sum(causal_attention_mask, [])[:-1]  # remove last eos
    
    
    situation_ids = situation_ids[-max_input_length:]
    input_ids = input_ids[-max_input_length:]
    causal_attention_mask = causal_attention_mask[-max_input_length:]
    strategy_history_input_ids = strategy_history[-max_input_length:]
    
    labels = (response + [eos])[:max_decoder_input_length]
    decoder_input_ids = [bos] + labels[:-1]
    
    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]
    assert len(causal_attention_mask) == len(input_ids), len(causal_attention_mask)
    assert sum(causal_attention_mask) > 0, causal_attention_mask
    
    return InputFeatures(
        dialog_id, situation_ids, speakers, input_ids,
        causal_attention_mask, strategy_history_input_ids,
        decoder_input_ids, labels, strategy_id
    )


def get_last_post_attention_mask(causal_attention_mask, speakers):
    last_post_idx = -1
    for i, speaker in enumerate(speakers):
        if speaker == -1 or last_post_idx == -1:
            last_post_idx = i
    res = causal_attention_mask.copy()
    res[last_post_idx] = [1] * len(causal_attention_mask[last_post_idx])
    return res


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    
    situation_ids: List = process(_norm(data['situation']))
    user_id = process('[User]')[0]
    dialog_id = data['dialog_id']
    dialog = data['dialog']
    
    inputs = []
    context = []
    speakers = []
    causal_attention_mask = []
    context_str = []
    strategy_history = [toker.bos_token_id]
    
    for i in range(len(dialog)):
        text: str = _norm(dialog[i]['text'])
        text_ids: List[int] = process(text)
        
        if dialog[i]['speaker'] == 'sys':
            strategy_id: int = process('[' + dialog[i]['strategy'] + ']')[0]
        
        if i > 0 and dialog[i]['speaker'] == 'sys':
            
            res = {
                'dialog_id': dialog_id,
                'situation_ids': situation_ids.copy(),
                'speakers': speakers.copy(),
                'context': context.copy(),
                'context_str': context_str.copy(),
                'causal_attention_mask': get_last_post_attention_mask(
                    causal_attention_mask,
                    speakers
                ),
                'strategy_history': strategy_history.copy(),
                'response_str': text,
                'response': text_ids,
                'strategy_id': strategy_id,
            }
            inputs.append(res)

        if dialog[i]['speaker'] == 'sys':
            text_ids = [strategy_id] + text_ids
        else:
            text_ids = [user_id] + text_ids

        is_cause = CAUSAL_DF.loc[f"esconv.dialog_id_{dialog_id}_utt_{i+1}"].preds
        causal_attention_mask = causal_attention_mask + [[is_cause]*len(text_ids)]
        context = context + [text_ids]
        context_str = context_str + [text]
        speakers.append(1 if dialog[i]['speaker'] == 'sys' else -1)
        if dialog[i]['speaker'] == 'sys':
            strategy_history = strategy_history + [strategy_id]
        
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
            ipt['dialog_id'], ipt['situation_ids'], ipt['speakers'], ipt['context'], max_input_length,
            ipt['causal_attention_mask'], ipt['strategy_history'],
            ipt['response'], max_decoder_input_length, ipt['strategy_id']
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
        situation_ids = pad_sequence([torch.tensor(f.situation_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        situation_attention_mask = pad_sequence([torch.tensor([1.] * f.situation_length, dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)
        speaker_utterance = pad_sequence([torch.tensor(f.speakers, dtype=torch.long) for f in features],
                                         batch_first=True,
                                         padding_value=pad)
        situation_features = []
        utterance_features = []
        # all 9 relations in COMET
        for i in range(len(COMET_RELATIONS)):
            situation_features.append(torch.stack([torch.tensor(f.situation_features[i]) for f in features]))
            utterance_features.append(pad_sequence([torch.tensor(f.utterance_features[i]) for f in features],
                            batch_first=True, padding_value=pad))

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1] * f.input_length,  dtype=torch.long) for f in features],
                          batch_first=True, padding_value=0)
        
        causal_attention_mask = pad_sequence(
            [torch.tensor(f.causal_attention_mask, dtype=torch.long) for f in features],
            batch_first=True, padding_value=0)
        
        strategy_history_input_ids = pad_sequence(
            [torch.tensor(f.strategy_history_input_ids, dtype=torch.long) for f in features],
            batch_first=True, padding_value=pad
        )
        strategy_history_attention_mask = pad_sequence(
            [torch.tensor([1] * f.strategy_history_input_length, dtype=torch.long) for f in features],
            batch_first=True, padding_value=0)
        
        effect_situation_attention_mask = torch.ones(batch_size, len(SEK_RELATIONS))
        effect_context_attention_mask = torch.stack(
            len(SEK_RELATIONS)*[(speaker_utterance == -1).long()],
            dim=2)
        effect_context_attention_mask = effect_context_attention_mask.flatten(start_dim=1, end_dim=2)
        encoder_attention_mask = torch.cat([
            situation_attention_mask,
            attention_mask,
            effect_situation_attention_mask,
            effect_context_attention_mask
        ], dim=1)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                              batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                              batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None
        
        strategy_id = torch.tensor([f.strategy_id for f in features], dtype=torch.long) - len(toker) + 8
        
        res = {
            'situation_ids': situation_ids,
            'situation_attention_mask': situation_attention_mask,
            'situation_features': situation_features,
            'utterance_features': utterance_features,

            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'causal_attention_mask': causal_attention_mask,
            'encoder_attention_mask': encoder_attention_mask,
            
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
            posts.append(ipt['context_str'][-1])
            references.append(toker.decode(ipt['response'], skip_special_tokens=True))
            sample_ids.append(sample_id)
    
            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
