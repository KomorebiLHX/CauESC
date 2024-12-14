# coding=utf-8

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

import torch
from transformers import (PreTrainedTokenizer, PreTrainedModel, PretrainedConfig)
from .PARAMS import STRATEGIES

class BaseModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.toker = None
    
    def tie_tokenizer(self, toker: PreTrainedTokenizer):
        self.toker = toker
        if len(self.toker) > self.toker.vocab_size:
            self.resize_token_embeddings(len(self.toker))

        toker.pad_token_id = toker.pad_token_id if toker.pad_token_id else toker.eos_token_id
        
        if hasattr(self, 'register_strategy'):
            self.register_strategy()
