import torch

from .attention import GrowingAttention
from .base import Growing
from .configuration import GrowingConfig
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

    def forward(self, x):
        x = self.attention(x)
        out = self.mlp(x)
        out = self.layer_norm(x + out)
        return out
