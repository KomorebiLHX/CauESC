# coding=utf-8
import random
from typing import Optional, Union, Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.blenderbot_small import BlenderbotSmallConfig
from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallAttention, \
    BlenderbotSmallEncoderLayer, BlenderbotSmallEncoder

from .attentions import _expand_mask


class BlenderbotSmallCausalEncoderLayer(BlenderbotSmallEncoderLayer):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.causal_attn = BlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.weight_fc = nn.Linear(2*config.d_model, 1)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            layer_head_mask: torch.FloatTensor,
            causal_attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False
    ) -> tuple[Union[Tensor, Any]]:

        residual = hidden_states
        causal_hidden_states = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        if causal_attention_mask is not None:
            causal_hidden_states, *_ = self.causal_attn(
                hidden_states=causal_hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            concat_hidden_states = torch.cat([hidden_states, causal_hidden_states], dim=-1)
            weight = nn.ReLU()(self.weight_fc(concat_hidden_states))
            hidden_states = weight * causal_hidden_states + (1-weight) * hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
    
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BlenderbotSmallCausalEncoder(BlenderbotSmallEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotSmallEncoderLayer`].

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
        self.layers = nn.ModuleList([BlenderbotSmallCausalEncoderLayer(config) for _ in range(config.encoder_layers)])
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            emotion_id=None,
            attention_mask=None,
            causal_attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        if emotion_id is not None:
            embed_emo = self.embed_emotion(emotion_id)[:, None, :]
            hidden_states = hidden_states + embed_emo
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # expand causal_attention_mask
        if causal_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            causal_attention_mask = _expand_mask(causal_attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
