# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration
from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallModel, BlenderbotSmallEncoder
from transformers.modeling_outputs import (BaseModelOutput,
                                           Seq2SeqModelOutput,
                                           Seq2SeqLMOutput,
                                           BaseModelOutputWithPastAndCrossAttentions)
from sklearn.metrics import accuracy_score

from .model_utils import BaseModel
from .outputs import BaseModelOutputWithPoolingAndStrategy
from .attentions import KeyValueAttention
from .executors import BlenderbotSmallDecoderWithSmallExecutor
from .PARAMS import ALPHA, SAMPLE, TEMPERATURE, STRATEGIES


class BlenderbotSmallCausalModel(BlenderbotSmallModel):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

        self.encoder = BlenderbotSmallEncoder(config, self.shared)
        self.decoder = BlenderbotSmallDecoderWithSmallExecutor(config, self.shared)
        
        # Initialize weights and apply final processing
        self.post_init()


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        
        self.model = BlenderbotSmallCausalModel(config)
        
        self.persona_fc = nn.Linear(config.d_comet, config.d_model)
        self.effect_fc = nn.Linear(config.d_comet, config.d_model)
        self.strategy_fc = nn.Linear(2*config.d_model, config.strategy_size, bias=False)
        
        self.situation_attn = KeyValueAttention(
            embed_dim=config.d_model,
            num_heads=config.situation_attention_heads,
            dropout=config.attention_dropout
        )
        self.persona_layer_norm = nn.LayerNorm(config.d_model)
        self.context_attn = KeyValueAttention(
            embed_dim=config.d_model,
            num_heads=config.context_attention_heads,
            dropout=config.attention_dropout
        )
        self.effect_layer_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = config.dropout
        
        # self.effect_situation_fc = self.persona_fc
        # self.effect_context_fc = self.effect_fc
        # self.effect_situation_layer_norm = self.persona_layer_norm
        # self.effect_context_layer_norm = self.effect_layer_norm
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def register_strategy(self):
        batch_encoding = self.toker(list(STRATEGIES.values()), padding=True, return_tensors='pt')
        self.register_buffer('strategy_input_ids', batch_encoding.input_ids)
        self.register_buffer('strategy_attention_mask', batch_encoding.attention_mask)
    
    def _pad_utterance(self, hidden_state, original_shape, real_utt_len):
        batch_size, utt_len, seq_len = original_shape
        d_model = hidden_state.size(-1)
        
        for i in range(batch_size):
            fullzeropad4insert = hidden_state.new_zeros(utt_len - real_utt_len[i], seq_len, d_model, dtype=torch.float)
            index4insert = i * utt_len + real_utt_len[i]
            hidden_state = torch.cat([
                hidden_state[:index4insert],
                fullzeropad4insert,
                hidden_state[index4insert:]
            ], dim=0)
        
        hidden_state = hidden_state.view(batch_size, utt_len*seq_len, d_model)
        return hidden_state
    
    def _pad_features(self, hidden_state, original_shape, real_utt_len): # [real_utt, 6, d_model]
        batch_size, utt_len, seq_len, d_model = original_shape # seq_len=6
        
        for i in range(batch_size):
            fullzeropad4insert = hidden_state.new_zeros(utt_len - real_utt_len[i], seq_len, d_model).float()
            index4insert = i * utt_len + real_utt_len[i]
            hidden_state = torch.cat([
                hidden_state[:index4insert],
                fullzeropad4insert,
                hidden_state[index4insert:]
            ], dim=0)
        
        hidden_state = hidden_state.view(batch_size, utt_len*seq_len, d_model)
        return hidden_state
        
    def get_encoder_outputs(
        self,
        input_ids,
        attention_mask,
        situation_ids,
        situation_attention_mask,
        situation_features,
        utterance_features,
        strategy_history_input_ids: torch.Tensor,
        strategy_history_attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # situation_hidden_state
        situation_encoder_outputs: BaseModelOutput = self.model.encoder(
            input_ids=situation_ids,
            attention_mask=situation_attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        situation_hidden_state = situation_encoder_outputs.last_hidden_state # torch.Size([bs, sit_len, dim])

        # comet features from situation
        x1, x2, x3, x4, x5, x6, o1, o2, o3 = situation_features # torch.Size([bs, dim])
        effect_situation_hidden_state: torch.Tensor = self.persona_fc(torch.stack([x4, x5, x6], dim=1)) # torch.Size([bs, 6, dim])
        
        # situation attention with add&norm
        residual = effect_situation_hidden_state
        effect_situation_features, *_ = self.situation_attn(
            hidden_states=effect_situation_hidden_state,
            key_value_states=situation_hidden_state,
            attention_mask=situation_attention_mask
        )
        effect_situation_features = F.dropout(effect_situation_features, p=self.dropout, training=self.training)
        effect_situation_features = residual + effect_situation_features
        effect_situation_features = self.persona_layer_norm(effect_situation_features)
        
        # context
        context_encoder_outputs: BaseModelOutput = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        context_hidden_state = context_encoder_outputs.last_hidden_state # torch.Size([bs, ctx_len, dim])
        # masked average pooling
        pooler_context = (torch.sum(context_hidden_state*attention_mask[..., None], dim=1) / 
                         torch.sum(attention_mask, dim=1)[:, None]) # torch.Size([bs, dim])
        
        # comet features from context
        x1, x2, x3, x4, x5, x6, o1, o2, o3 = utterance_features # torch.Size([bs, utt_len, dim])
        effect_context_hidden_state: torch.Tensor = self.effect_fc(torch.stack([x4, x5, x6, o1, o2, o3], dim=2)) # torch.Size([bs, utt_len, seq_len, d_model])
        effect_context_hidden_state = effect_context_hidden_state.flatten(start_dim=1, end_dim=2) # torch.Size([bs, utt_len*seq_len, d_model])
        
        # context attention with add&norm
        residual = effect_context_hidden_state
        effect_context_features, *_ = self.context_attn(
            hidden_states=effect_context_hidden_state,
            key_value_states=context_hidden_state,
            attention_mask=attention_mask
        )
        effect_context_features = F.dropout(effect_context_features, p=self.dropout, training=self.training)
        effect_context_features = residual + effect_context_features
        effect_context_features = self.effect_layer_norm(effect_context_features)
        
        # total with norm
        last_hidden_state: torch.Tensor = torch.cat([
            situation_hidden_state,
            context_hidden_state,
            effect_situation_features,
            effect_context_features
        ], dim=1)

        # strategy history
        strategy_history_hidden_state = self.model.encoder(
            input_ids=strategy_history_input_ids,
            attention_mask=strategy_history_attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ).last_hidden_state  # torch.Size([bs, ctx_len, dim])
        pooler_strategy_history = (torch.sum(strategy_history_hidden_state*strategy_history_attention_mask[..., None], dim=1)
                                   / torch.sum(strategy_history_attention_mask, dim=1)[:, None])

        # strategy hidden states
        strategy_hidden_states: torch.FloatTensor = self.model.encoder(
            input_ids=self.strategy_input_ids,
            attention_mask=self.strategy_attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ).last_hidden_state # [8, len, dim]

        pooler_output = torch.cat([pooler_context, pooler_strategy_history], dim=-1)  # torch.Size([bs, dim])
        strategy_logits = self.strategy_fc(pooler_output)
        strategy_score = F.softmax(strategy_logits, dim=-1) # torch.Size([bs, 8])
        
        encoder_outputs = BaseModelOutputWithPoolingAndStrategy(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            strategy_hidden_states=strategy_hidden_states,
            strategy_score=strategy_score
        )
        
        return encoder_outputs
    
    def forward(
        self,
        encoder_attention_mask,
        decoder_input_ids,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        situation_ids: Optional[torch.LongTensor] = None,
        situation_attention_mask: Optional[torch.LongTensor] = None,
        situation_features: Optional[List[torch.FloatTensor]] = None,
        utterance_features: Optional[List[torch.FloatTensor]] = None,
        strategy_history_input_ids: Optional[torch.Tensor] = None,
        strategy_history_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        strategy_id: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        validation: Optional[bool] = False,
        **kwargs
    ):
        assert self.toker is not None
        assert (self.training or validation) == (labels is not None)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs: BaseModelOutputWithPoolingAndStrategy = self.get_encoder_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_history_input_ids = strategy_history_input_ids,
                strategy_history_attention_mask = strategy_history_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                situation_ids=situation_ids,
                situation_attention_mask=situation_attention_mask,
                situation_features=situation_features,
                utterance_features=utterance_features,
            )

        if self.training or validation:
            ce_loss = F.cross_entropy(encoder_outputs.strategy_score, strategy_id)
            pred_strategy_id = torch.argmax(encoder_outputs.strategy_score, dim=-1)
            acc = accuracy_score(strategy_id.cpu().numpy(), pred_strategy_id.cpu().numpy())

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.model.decoder(
            executor_score=encoder_outputs.strategy_score,
            strategy_hidden_states=encoder_outputs.strategy_hidden_states,
            strategy_attention_mask=self.strategy_attention_mask,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        
        if not self.training and not validation: # inference
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions
            )

        elif self.training: # training
            step, num_optim_steps = kwargs['global_step'], kwargs['args'].num_optim_steps
            rate = step / num_optim_steps
            all_loss = ce_loss
            if rate >= ALPHA:
                all_loss = ce_loss + masked_lm_loss
            
            res = {
                'all': all_loss,
                'ppl': ppl_value,
                'acc': acc
            }
            return res

        else: # validation
            return loss, label_size, pred_strategy_id, strategy_id
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        encoder_attention_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "encoder_attention_mask": encoder_attention_mask
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        situation_ids,
        situation_attention_mask,
        encoder_attention_mask,
        situation_features,
        utterance_features,
        strategy_history_input_ids,
        strategy_history_attention_mask,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs: BaseModelOutputWithPoolingAndStrategy = self.get_encoder_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_history_input_ids = strategy_history_input_ids,
                strategy_history_attention_mask = strategy_history_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                situation_ids=situation_ids,
                situation_attention_mask=situation_attention_mask,
                situation_features=situation_features,
                utterance_features=utterance_features,
            )

        strategy_id = encoded_info.get('strategy_id', None)
        if strategy_id is not None: # if given stategy, use golden truth
            encoder_outputs.strategy_score = F.one_hot(strategy_id, num_classes=self.config.strategy_size).float()
            strategy_score = encoder_outputs.strategy_score
            pred_strategy = strategy_id
        else: # use predicted strategy
            strategy_score = encoder_outputs.strategy_score
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(strategy_score / TEMPERATURE, top_p=0.9)
                pred_strategy = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred_strategy = torch.argmax(strategy_score, dim=-1)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids

        other_res = encoded_info.pop('other_res')
        generations = super().generate(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        encoded_info['other_res'] = other_res
        encoded_info.update({
            'pred_strategy_id': pred_strategy,
            'pred_strategy_id_top1': torch.topk(strategy_score, k=1, dim=-1)[1],
            'pred_strategy_id_top3': torch.topk(strategy_score, k=3, dim=-1)[1],
            'pred_strategy_id_dist': strategy_score
        })

        return encoded_info, generations[:, decoder_input_ids.size(1):]
