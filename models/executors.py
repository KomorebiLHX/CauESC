# coding=utf-8
import logging
import math
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.blenderbot_small import BlenderbotSmallConfig
from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallDecoder, BlenderbotSmallDecoderLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from models.attentions import _expand_mask, KeyValueAttention

logger = logging.getLogger(__name__)


class SmallExecutor(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.strategy_size = config.strategy_size
        self.attentions = nn.ModuleList([KeyValueAttention(
            embed_dim=config.d_model,
            num_heads=config.context_attention_heads,
            dropout=config.attention_dropout
        ) for _ in range(config.strategy_size)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        executor_score: torch.Tensor,
        strategy_hidden_states: torch.Tensor,
        strategy_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs = hidden_states.size(0)
        executor_score = executor_score[..., None, None]  # torch.Size([bs, 8, 1, 1])
        strategy_hidden_states = strategy_hidden_states[None, ...].expand(bs, -1, -1, -1)  # torch.Size([bs, 8, len, dim])
        strategy_attention_mask = strategy_attention_mask[None, ...].expand(bs, -1, -1)  # torch.Size([bs, 8, len])
        encoder_hidden_states = encoder_hidden_states[:, None, ...].expand(-1, self.strategy_size, -1, -1)  # torch.Size([bs, 8, len, dim])
        encoder_attention_mask = encoder_attention_mask[:, None, ...].expand(-1, self.strategy_size, -1)  # torch.Size([bs, 8, len])
        concat_hidden_states = torch.cat([encoder_hidden_states, strategy_hidden_states], dim=2)
        concat_attention_mask = torch.cat([encoder_attention_mask, strategy_attention_mask], dim=2)
        
        residual = hidden_states
        attention_outputs = ()
        for idx, attention in enumerate(self.attentions):
            attention_output, *_ = attention(
                hidden_states=hidden_states,
                key_value_states=concat_hidden_states[:, idx],
                attention_mask=concat_attention_mask[:, idx],
            )
            attention_outputs += (attention_output,)
        stacked_attention_outputs = torch.stack([output for output in attention_outputs], dim=1) # torch.Size([bs, 8, len, dim])
        pooler_output = torch.sum(executor_score * stacked_attention_outputs, dim=1)
        pooler_output = F.dropout(pooler_output, p=self.dropout, training=self.training)
        pooler_output = residual + pooler_output
        pooler_output = self.final_layer_norm(pooler_output)

        return pooler_output


class Executor(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.strategy_size = config.strategy_size
        self.executor_layers = nn.ModuleList(
            [BlenderbotSmallDecoderLayer(config) for _ in range(config.strategy_size)]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        executor_score: torch.Tensor,
        strategy_hidden_states: torch.Tensor,
        strategy_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        bs = hidden_states.size(0)
        executor_score = executor_score[..., None, None]  # torch.Size([bs, 8, 1, 1])
        strategy_hidden_states = strategy_hidden_states[None, ...].expand(bs, -1, -1, -1)  # torch.Size([bs, 8, len, dim])
        strategy_attention_mask = strategy_attention_mask[None, ...].expand(bs, -1, -1)  # torch.Size([bs, 8, len])
        encoder_hidden_states = encoder_hidden_states[:, None, ...].expand(-1, self.strategy_size, -1, -1)  # torch.Size([bs, 8, len, dim])
        encoder_attention_mask = encoder_attention_mask[:, None, ...].expand(-1, self.strategy_size, -1)  # torch.Size([bs, 8, len])
        concat_hidden_states = torch.cat([encoder_hidden_states, strategy_hidden_states], dim=2)
        concat_attention_mask = torch.cat([encoder_attention_mask, strategy_attention_mask], dim=2)

        residual = hidden_states
        layer_outputs = ()
        for idx, layer in enumerate(self.executor_layers):
            layer_output, *_ = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=concat_hidden_states[:, idx],
                encoder_attention_mask=_expand_mask(
                    concat_attention_mask[:, idx],
                    hidden_states.dtype,
                    tgt_len=hidden_states.size(1)
                    ) # expand strategy attention mask
            )
            layer_outputs += (layer_output,)
        stacked_layer_outputs = torch.stack([output for output in layer_outputs], dim=1) # torch.Size([bs, 8, len, dim])
        pooler_output = torch.sum(executor_score * stacked_layer_outputs, dim=1)
        pooler_output = F.dropout(pooler_output, p=self.dropout, training=self.training)
        pooler_output = residual + pooler_output
        pooler_output = self.final_layer_norm(pooler_output)

        return pooler_output


class BlenderbotSmallDecoderWithExecutor(BlenderbotSmallDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotSmallDecoderLayer`]

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
            self,
            config: BlenderbotSmallConfig,
            embed_tokens: Optional[nn.Embedding] = None,
            embed_emotion: Optional[nn.Embedding] = None
    ):
        super().__init__(config, embed_tokens)
        self.embed_emotion = embed_emotion
        self.executor = Executor(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        executor_score=None,
        strategy_hidden_states=None,
        strategy_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        emotion_id=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        # BlenderbotSmall applies layer norm on hidden_states
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        hidden_states = inputs_embeds + positions
        if emotion_id is not None:
            embed_emo = self.embed_emotion(emotion_id)[:, None, :]
            hidden_states = hidden_states + embed_emo

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # executor
        hidden_states = self.executor(
            hidden_states=hidden_states,
            executor_score=executor_score,
            attention_mask=attention_mask,
            strategy_hidden_states=strategy_hidden_states,
            strategy_attention_mask=strategy_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BlenderbotSmallDecoderWithSmallExecutor(BlenderbotSmallDecoderWithExecutor):
    def __init__(
            self,
            config: BlenderbotSmallConfig,
            embed_tokens: Optional[nn.Embedding] = None,
            embed_emotion: Optional[nn.Embedding] = None
    ):
        super().__init__(config, embed_tokens, embed_emotion)
        self.executor = SmallExecutor(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            executor_score=None,
            strategy_hidden_states=None,
            strategy_attention_mask=None,
            input_ids=None,
            attention_mask=None,
            emotion_id=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        # BlenderbotSmall applies layer norm on hidden_states
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        hidden_states = inputs_embeds + positions
        if emotion_id is not None:
            embed_emo = self.embed_emotion(emotion_id)[:, None, :]
            hidden_states = hidden_states + embed_emo

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # executor
        hidden_states = self.executor(
            hidden_states=hidden_states,
            executor_score=executor_score,
            attention_mask=attention_mask,
            strategy_hidden_states=strategy_hidden_states,
            strategy_attention_mask=strategy_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )