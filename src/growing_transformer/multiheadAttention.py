import torch
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm
from torch.nn.init import uniform_

from contextlib import contextmanager
from typing import Optional, List, Iterable, OrderedDict, Tuple, Union, Mapping, Any

from . import GrowingModule, ScaledDotProductAttention


_bert_state_dict_map = {
    'self.query.weight': 'dot_product.query_linear.weight',
    'self.query.bias': 'dot_product.query_linear.bias',
    'self.key.weight': 'dot_product.key_linear.weight',
    'self.key.bias': 'dot_product.key_linear.bias',
    'self.value.weight': 'value_linear.weight',
    'self.value.bias': 'value_linear.bias',
    'output.dense.weight': 'output_linear.weight',
    'output.dense.bias': 'output_linear.bias',
    'output.LayerNorm.weight': 'layer_norm.weight',
    'output.LayerNorm.bias': 'layer_norm.bias',
}


class MultiheadAttention(GrowingModule):
    def __init__(self, d_model: int, heads: int, d_head: int, config: Mapping[str, Any] = {}, bert_state_dict=False):
        super().__init__(config)

        self.dot_product = ScaledDotProductAttention(d_model, heads, d_head)

        self.value_linear = Linear(d_model, heads*d_head)
        self.output_linear = Linear(heads*d_head, d_model)

        self.layer_norm = LayerNorm(d_model, eps=config.get('layer_norm_eps', 1e-12))

        self.heads = heads
        self.d_head = d_head
        self.d_model = d_model

        if bert_state_dict:
            self._register_state_dict_hook(self._bert_state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_bert_state_dict_pre_hook)

        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.new_neurons = None

        # update directions (to be trained) after growing
        self._value_weight: Optional[torch.nn.Parameter] = None
        self._value_bias: Optional[torch.nn.Parameter] = None
        self._output_weight: Optional[torch.nn.Parameter] = None

    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        return [
            self._value_weight,
            self._value_bias,
            self._output_weight,
        ]

    @property
    def in_features(self) -> int:
        return self.d_model

    def forward(self,
            x: Tensor,
            return_attention: bool = False
            ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        batch, length, _ = x.size()

        attention = self.dot_product(x)

        value = self.value_linear(x).view(batch, length, self.heads, self.d_head)

        einsum_str = 'bhqk,bkhe->bqhe'
        out = torch.einsum(einsum_str, attention, value).reshape(batch, length, -1)

        out = self.output_linear(out)

        if self.new_neurons is not None:
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            value_novel = torch.nn.functional.linear(x, self._value_weight, self._value_bias)
            value_novel = value_novel.view(batch, length, self.heads, -1)

            out_novel = torch.einsum(einsum_str, attention, value_novel)
            out_novel = out_novel * self.new_neurons[None, None, :, :]
            out_novel = out_novel.reshape(batch, length, -1)

            # linear transform (like output_linear)
            out_novel = torch.nn.functional.linear(out_novel, self._output_weight)

            out += out_novel

        out = self.layer_norm(out + x)

        if not return_attention:
            return out
        else:
            return out, attention

    def grow(self, step_size: float = 1e-1) -> torch.Size:
        num_novel = self.get_config('num_novel', default=0)
        eps_novel_weight = self.get_config('eps_novel_weight', 'eps_novel', default=1e-1)
        eps_novel_bias =self.get_config('eps_novel_bias', 'eps_novel', default=1e-1)

        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = Parameter(
            torch.ones(self.heads, num_novel) * step_size,
            requires_grad=False
        )

        # create update direction for weight and bias
        self._output_weight = None

        self._value_weight = Parameter(torch.empty(self.heads * num_novel, self.d_model), requires_grad=False)
        self._value_bias = Parameter(torch.empty(self.heads * num_novel), requires_grad=False)
        self._output_weight = Parameter(torch.empty(self.d_model, self.heads * num_novel), requires_grad=False)

        uniform_(self._value_weight, -eps_novel_weight, eps_novel_weight)
        uniform_(self._output_weight, -eps_novel_weight, eps_novel_weight)
        uniform_(self._value_bias, -eps_novel_bias, eps_novel_bias)

        return self.new_neurons.size()

    def degrow(self, selected: Tensor) -> None:
        with torch.no_grad():

            if selected.size(0) == 0:
                self.reset_grow_state()
                return

            assert selected.size(0) % self.heads == 0

            assert self.new_neurons is not None
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            d_new = selected.size(0) // self.heads

            value_weight = torch.empty(self.heads, self.d_head + d_new, self.d_model)
            value_bias = torch.empty(self.heads, self.d_head + d_new)

            output_weight = torch.empty(self.d_model, self.heads, self.d_head + d_new)

            # copy old neurons

            value_weight[:, :self.d_head] = self.value_linear.weight.view(self.heads, self.d_head, self.d_model)
            value_bias[:, :self.d_head] = self.value_linear.bias.view(self.heads, self.d_head)

            output_weight[..., :self.d_head] = self.output_linear.weight.view(self.d_model, self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.new_neurons.view(-1)[selected].view(self.heads, -1)

            value_weight[:, self.d_head:, :] = (
                self._value_weight.view(-1, self.d_model)[selected].view(self.heads, -1, self.d_model)
                * selected_steps[:, :, None]
            )
            value_bias[:, self.d_head:] = self._value_bias.view(-1)[selected].view(self.heads, -1) * selected_steps

            output_weight[..., self.d_head:] = self._output_weight.view(self.d_model, -1)[:, selected].view(self.d_model, self.heads, -1)

            self.d_head = self.d_head + d_new

            self.value_linear.weight = torch.nn.Parameter(value_weight.reshape(self.heads*self.d_head, self.d_model))
            self.value_linear.bias = torch.nn.Parameter(value_bias.reshape(self.heads*self.d_head))
            self.output_linear.weight = torch.nn.Parameter(output_weight.reshape(self.d_model, self.heads*self.d_head))

        self.reset_grow_state()

    @staticmethod
    def _bert_state_dict_hook(self, state_dict, prefix, local_metadata):
        for k,v in _bert_state_dict_map.items():
            state_dict[prefix + k] = state_dict[prefix + v]
            del state_dict[prefix + v]

    def _load_bert_state_dict_pre_hook(self, state_dict, prefix, *_):
        for k,v in _bert_state_dict_map.items():
            state_dict[prefix + v] = state_dict[prefix + k]
            del state_dict[prefix + k]
