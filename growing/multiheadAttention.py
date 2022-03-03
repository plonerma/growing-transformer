import torch

from contextlib import contextmanager
from typing import Optional

from .base import GrowingModule
from . import ScaledDotProductAttention
from .util import map_attention_state


class MultiheadAttention(GrowingModule):
    def __init__(self, d_model, heads, d_head, config={}):
        super().__init__()

        self.dot_product = ScaledDotProductAttention(d_model, heads, d_head)

        self.value_linear = torch.nn.Linear(d_model, heads*d_head)
        self.output_linear = torch.nn.Linear(heads*d_head, d_model)

        self.layer_norm = torch.nn.LayerNorm(d_model, eps=config.get('layer_norm_eps', 1e-12))

        self.heads = heads
        self.d_head = d_head
        self.d_model = d_model

    def reset_grow_state(self):
        super().reset_grow_state()

        # update directions (to be trained)
        self._weight_dir = None
        self._bias_dir = None

    def direction_params(self):
        return [
            self._weight_dir,
            self._bias_dir
        ]

    def state_dict(self, bert_like=False):
        state = super().state_dict()

        if bert_like:
            state = map_attention_state(state, from_bert=False)

        return state

    def load_state_dict(self, state, bert_like=False):
        if bert_like:
            state = map_attention_state(state, from_bert=True)
        super().load_state_dict(state)

    @property
    def in_features(self):
        return self.d_model

    def forward(self, x, return_attention=False):
        batch, length, _ = x.size()

        attention = self.dot_product(x)

        value = self.value_linear(x).view(batch, length, self.heads, self.d_head)

        print(attention.size(), value.size())

        out = torch.einsum('bhqk,bkhe->bqhe', attention, value).reshape(batch, length, -1)

        print("attention out", out)

        out = self.output_linear(out)

        print("linear out", out)

        out = self.layer_norm(out + x)

        print("layer norm out", out)

        if not return_attention:
            return out
        else:
            return out, attention

    def grow(self,
             num_novel : int = 0,
             step_size = 1,
             eps_novel : float = 1e-2,
             eps_novel_weight : Optional[float] = None,
             eps_novel_bias : Optional[float] = None,
             **kw):
        raise NotImplementedError

    def degrow(self, selected : torch.Tensor):
        raise NotImplementedError
