from typing import Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Parameter
from torch.nn.init import uniform_

from ..configuration import GrowingConfig
from .base import GrowingModule


class GrowingAttention(GrowingModule):
    _bert_state_dict_map = {
        "self.query": "dot_product.query_linear",
        "self.key": "dot_product.key_linear",
        "self.value": "value_linear",
        "output.dense": "output_linear",
        "output.LayerNorm": "layer_norm",
    }

    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)

        self.heads = config.num_attention_heads
        self.d_head = config.d_head_v
        self.d_model = config.d_model

        self.dot_product = ScaledDotProductAttention(config=config)

        self.value_linear = Linear(self.d_model, self.heads * self.d_head)
        self.output_linear = Linear(self.heads * self.d_head, self.d_model)

        self.layer_norm = LayerNorm(self.d_model, eps=config.layer_norm_eps)

        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.new_parts = None

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

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        influence_factor=1.0,
        attention_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        batch, length, _ = x.size()

        attention = self.dot_product(x, attention_mask=attention_mask)

        value = self.value_linear(x).view(batch, length, self.heads, self.d_head)

        einsum_str = "bhqk,bkhe->bqhe"
        out = torch.einsum(einsum_str, attention, value).reshape(batch, length, -1)

        out = self.output_linear(out)

        if self.new_parts is not None:
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            value_novel = torch.nn.functional.linear(x, self._value_weight, self._value_bias)
            value_novel = value_novel.view(batch, length, self.heads, -1)

            out_novel = torch.einsum(einsum_str, attention, value_novel)
            out_novel = out_novel * self.new_parts[None, None, :, :]
            out_novel = out_novel.reshape(batch, length, -1)

            # linear transform (like output_linear)
            out_novel = torch.nn.functional.linear(out_novel, self._output_weight)

            out += out_novel

        out = out * influence_factor
        out = self.layer_norm(x + out)

        if not return_attention:
            return out
        else:
            return out, attention

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        step_size = self.config.step_size
        eps_novel_weight = self.config.eps_novel_weight
        eps_novel_bias = self.config.eps_novel_bias

        # add parameter to measure influence/gradient of adding new neurons
        self.new_parts = Parameter(torch.ones(self.heads, num_novel) * step_size, requires_grad=False)

        # create update direction for weight and bias
        self._output_weight = None

        self._value_weight = Parameter(torch.empty(self.heads * num_novel, self.d_model), requires_grad=False)
        self._value_bias = Parameter(torch.empty(self.heads * num_novel), requires_grad=False)
        self._output_weight = Parameter(torch.empty(self.d_model, self.heads * num_novel), requires_grad=False)

        uniform_(self._value_weight, -eps_novel_weight, eps_novel_weight)
        uniform_(self._output_weight, -eps_novel_weight, eps_novel_weight)
        uniform_(self._value_bias, -eps_novel_bias, eps_novel_bias)

        return self.new_parts.size()

    def degrow(self, selected: Tensor) -> None:
        with torch.no_grad():

            if selected.size(0) == 0:
                self.reset_grow_state()
                return

            assert selected.size(0) % self.heads == 0

            assert self.new_parts is not None
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            d_new = selected.size(0) // self.heads

            value_weight = torch.empty(self.heads, self.d_head + d_new, self.d_model)
            value_bias = torch.empty(self.heads, self.d_head + d_new)

            output_weight = torch.empty(self.d_model, self.heads, self.d_head + d_new)

            # copy old neurons

            value_weight[:, : self.d_head] = self.value_linear.weight.view(self.heads, self.d_head, self.d_model)
            value_bias[:, : self.d_head] = self.value_linear.bias.view(self.heads, self.d_head)

            output_weight[..., : self.d_head] = self.output_linear.weight.view(self.d_model, self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.new_parts.view(-1)[selected].view(self.heads, -1)

            value_weight[:, self.d_head :, :] = (
                self._value_weight.view(-1, self.d_model)[selected].view(self.heads, -1, self.d_model)
                * selected_steps[:, :, None]
            )
            value_bias[:, self.d_head :] = self._value_bias.view(-1)[selected].view(self.heads, -1) * selected_steps

            output_weight[..., self.d_head :] = self._output_weight.view(self.d_model, -1)[:, selected].view(
                self.d_model, self.heads, -1
            )

            self.d_head = self.d_head + d_new

            self.value_linear.weight = torch.nn.Parameter(value_weight.reshape(self.heads * self.d_head, self.d_model))
            self.value_linear.bias = torch.nn.Parameter(value_bias.reshape(self.heads * self.d_head))
            self.output_linear.weight = torch.nn.Parameter(
                output_weight.reshape(self.d_model, self.heads * self.d_head)
            )

        self.reset_grow_state()
        self.config.d_head_v = self.d_head


class ScaledDotProductAttention(GrowingModule):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)

        self.heads = config.num_attention_heads
        self.d_head = config.d_head_kq
        self.query_linear = torch.nn.Linear(config.d_model, self.heads * self.d_head)
        self.key_linear = torch.nn.Linear(config.d_model, self.heads * self.d_head)
        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.new_parts = None

        # update directions (to be trained)
        self._weight: Optional[torch.nn.Parameter] = None
        self._bias: Optional[torch.nn.Parameter] = None

    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        return [
            self._weight,
            self._bias,
        ]

    @property
    def d_model(self) -> int:
        return self.query_linear.weight.size(1)

    @property
    def in_features(self) -> int:
        return self.d_model

    def forward(self, x: torch.Tensor, attention_mask: Optional[Tensor] = None) -> torch.Tensor:
        batch_size, length, _ = x.size()

        q = self.query_linear(x).view(batch_size, length, self.heads, -1)

        k = self.key_linear(x).view(batch_size, length, self.heads, -1)

        # Batch (b) and head (h) dimensions remain intact
        # Query dimension (q) remains in the same place
        # Embedding values (e) are used in actual dot product
        einsum_str = "bqhe,bkhe->bhqk"
        product: torch.Tensor = torch.einsum(einsum_str, q, k)

        if self.new_parts is not None:
            assert self._weight is not None and self._bias is not None

            q_novel = torch.nn.functional.linear(x, self._weight[0], self._bias[0])
            q_novel = q_novel.view(batch_size, length, self.heads, -1)
            q_novel = q_novel * self.new_parts

            k_novel = torch.nn.functional.linear(x, self._weight[1], self._bias[1])
            k_novel = k_novel.view(batch_size, length, self.heads, -1)

            product = product + torch.einsum(einsum_str, q_novel, k_novel)

        product = product / torch.sqrt(torch.tensor(self.d_head))

        if attention_mask is not None:
            product = product.masked_fill(
                # apply mask to every head and every query token
                attention_mask[:, None, None, :].bool(),
                # fill with very small value
                -10000,
            )

        return torch.softmax(product, dim=-1)

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        step_size = self.config.step_size
        eps_novel_weight = self.config.eps_novel_weight
        eps_novel_bias = self.config.eps_novel_bias

        # add parameter to measure influence/gradient of adding new neurons
        self.new_parts = torch.nn.Parameter(torch.ones(self.heads, num_novel) * step_size, requires_grad=False)

        # create update direction for weight and bias
        self._weight = torch.nn.Parameter(torch.empty(2, self.heads * num_novel, self.d_model), requires_grad=False)

        self._bias = torch.nn.Parameter(torch.empty(2, self.heads * num_novel), requires_grad=False)

        torch.nn.init.uniform_(self._weight, -eps_novel_weight, eps_novel_weight)

        torch.nn.init.uniform_(self._bias, -eps_novel_bias, eps_novel_bias)

        return self.new_parts.size()

    def degrow(self, selected: torch.Tensor) -> None:
        with torch.no_grad():

            if self.new_parts is None:
                return

            assert self._weight is not None
            assert self._bias is not None

            d_new = selected.size(0) // self.heads

            q_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features)

            k_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features)

            q_bias = torch.empty(self.heads, self.d_head + d_new)

            k_bias = torch.empty(self.heads, self.d_head + d_new)

            # copy old neurons

            q_weight[:, : self.d_head] = self.query_linear.weight.view(self.heads, self.d_head, self.d_model)
            k_weight[:, : self.d_head] = self.key_linear.weight.view(self.heads, self.d_head, self.d_model)
            q_bias[:, : self.d_head] = self.query_linear.bias.view(self.heads, self.d_head)
            k_bias[:, : self.d_head] = self.key_linear.bias.view(self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.new_parts.view(-1)[selected].view(self.heads, -1)

            # temporarily merge heads and output dimension
            selected_weight = self._weight.view(2, -1, self.d_model)
            # only use selected neurons
            selected_weight = selected_weight[:, selected]
            # split up dimensions again
            selected_weight = selected_weight.view(2, self.heads, -1, self.d_model)

            # same for bias
            selected_bias = self._bias.view(2, -1)[:, selected].view(2, self.heads, -1)

            q_weight[:, self.d_head :] = selected_weight[0] * selected_steps[..., None]
            k_weight[:, self.d_head :] = selected_weight[1]
            q_bias[:, self.d_head :] = selected_bias[0] * selected_steps
            k_bias[:, self.d_head :] = selected_bias[1]

            scale = torch.sqrt(torch.tensor(1 + d_new / self.d_head))

            q_weight *= scale
            q_bias *= scale

            self.d_head = self.d_head + d_new

            self.query_linear.weight = torch.nn.Parameter(q_weight.reshape(self.heads * self.d_head, self.d_model))
            self.key_linear.weight = torch.nn.Parameter(k_weight.reshape(self.heads * self.d_head, self.d_model))
            self.query_linear.bias = torch.nn.Parameter(q_bias.reshape(self.heads * self.d_head))
            self.key_linear.bias = torch.nn.Parameter(k_bias.reshape(self.heads * self.d_head))

        self.reset_grow_state()
        self.config.d_head_kq = self.d_head
