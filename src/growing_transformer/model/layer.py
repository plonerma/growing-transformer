from typing import Optional

import torch
from torch import Tensor

from ..configuration import GrowingConfig
from .attention import GrowingAttention
from .base import Growing
from .mlp import GrowingMLP as MLP


class GrowingLayer(Growing):
    _bert_state_dict_map = {
        "intermediate.dense": "mlp.linear_in",
        "output.dense": "mlp.linear_out",
        "output.LayerNorm": "layer_norm",
    }

    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)

        self.attention = GrowingAttention(config=config)
        self.mlp = MLP(config=config)

        eps = self.config.layer_norm_eps
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=eps)

        if self.config.bert_like_state_dict:
            self._register_state_dict_hook(self._bert_state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_bert_state_dict_pre_hook)

    def forward(self, x: Tensor, influence_factor=1.0, attention_mask: Optional[Tensor] = None):
        x1 = self.attention(x, influence_factor=influence_factor, attention_mask=attention_mask)
        mlp_out = self.mlp(x1) * influence_factor
        x2 = self.layer_norm(x1 + mlp_out)
        return x2

    def apply_influence_factor(self, f):
        self.mlp.linear_out.weight.data *= f
        self.mlp.linear_out.bias.data *= f
        self.attention.output_linear.weight.data *= f
        self.attention.output_linear.bias.data *= f
