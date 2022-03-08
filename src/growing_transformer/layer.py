from typing import Any, Mapping

import torch
from transformers.activations import ACT2FN

from .base import Growing
from .mlp import GrowingMLP as MLP
from .multiheadAttention import GrowingMultiheadAttention as MultiheadAttention

_bert_state_dict_map = {
    "intermediate.dense": "mlp.linear_in",
    "output.dense": "mlp.linear_out",
    "output.LayerNorm": "layer_norm",
}


class GrowingLayer(Growing):
    def __init__(self, d_model: int, heads: int, d_head: int, hidden_size: int, *, config: Mapping[str, Any]):
        super().__init__(config)

        intermediate_act_fn = self.get_config("intermediate_act_fn", default="relu")

        if isinstance(intermediate_act_fn, str):
            intermediate_act_fn = ACT2FN[intermediate_act_fn]
        else:
            intermediate_act_fn = intermediate_act_fn

        self.attention = MultiheadAttention(d_model, heads, d_head, config=config)
        self.mlp = MLP(d_model, d_model, hidden_size, activation=intermediate_act_fn, config=config)

        eps = self.get_config("layer_norm_eps", default=1e-12)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=eps)

        if self.get_config("bert_state_dict", default=False):
            self._register_state_dict_hook(self._bert_state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_bert_state_dict_pre_hook)

    def forward(self, x):
        x = self.attention(x)
        out = self.mlp(x)
        out = self.layer_norm(x + out)
        return out

    @staticmethod
    def _bert_state_dict_hook(self, state_dict, prefix, local_metadata):
        for k, v in list(state_dict.items()):
            for k_new_prefix, k_prefix in _bert_state_dict_map.items():
                k_prefix = prefix + k_prefix
                k_new_prefix = prefix + k_new_prefix

                if k.startswith(k_prefix):
                    k_new = k_new_prefix + k[len(k_prefix) :]
                    state_dict[k_new] = state_dict[k]
                    del state_dict[k]
                    break

    def _load_bert_state_dict_pre_hook(self, state_dict, prefix, *_):
        for k, v in list(state_dict.items()):
            for k_prefix, k_new_prefix in _bert_state_dict_map.items():
                k_prefix = prefix + k_prefix
                k_new_prefix = prefix + k_new_prefix

                if k.startswith(k_prefix):
                    k_new = k_new_prefix + k[len(k_prefix) :]
                    state_dict[k_new] = state_dict[k]
                    del state_dict[k]
                    break
