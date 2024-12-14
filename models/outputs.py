from typing import Optional, Tuple
import torch
from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPooling


@dataclass
class BaseModelOutputWithPoolingAndStrategy(BaseModelOutputWithPooling):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    strategy_hidden_states: torch.FloatTensor = None
    strategy_score: torch.FloatTensor = None
