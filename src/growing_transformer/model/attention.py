from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Parameter

import growing_transformer

from ..configuration import GrowingConfig, first_not_none
from .base import GrowingModule, NamedDirectionParams, truncated_normal_


class AttentionOutput(GrowingModule):
    def __init__(self, config: GrowingConfig, heads: int = None, bias=True):
        super().__init__(config=config)

        self.heads = first_not_none(heads, config.num_attention_heads)
        self.d_head = config.d_head_v
        self.d_model = config.d_model

        self.value_linear = Linear(self.d_model, self.heads * self.d_head)
        self.output_linear = Linear(self.heads * self.d_head, self.d_model, bias=bias)

        self.reset_grow_state()
        self.to(growing_transformer.device)

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.step_size = None

        # update directions (to be trained) after growing
        self._value_weight: Optional[Parameter] = None
        self._value_bias: Optional[Parameter] = None
        self._output_weight: Optional[Parameter] = None

    def _direction_params(self) -> NamedDirectionParams:
        return {
            "value_direction_weight": self._value_weight,
            "value_direction_bias": self._value_bias,
            "output_direction_weight": self._output_weight,
        }

    @property
    def in_features(self) -> int:
        return self.d_model

    def forward(
        self,
        x: Tensor,
        attention: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        batch, length, _ = x.size()

        value = self.value_linear(x).view(batch, length, self.heads, self.d_head)

        einsum_str = "bhqk,bkhe->bqhe"
        out = torch.einsum(einsum_str, attention, value).reshape(batch, length, -1)

        out = self.output_linear(out)

        if self.step_size is not None:
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            value_novel = torch.nn.functional.linear(x, self._value_weight, self._value_bias)
            value_novel = value_novel.view(batch, length, self.heads, -1) * self.step_size[None, None, :, :]

            out_novel = torch.einsum(einsum_str, attention, value_novel)
            out_novel = out_novel
            out_novel = out_novel.reshape(batch, length, -1)

            # linear transform (like output_linear)
            out_novel = torch.nn.functional.linear(out_novel, self._output_weight)

            out += out_novel

        return out

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        # add parameter to measure influence/gradient of adding new neurons
        self.step_size = Parameter(
            torch.ones(self.heads, num_novel, device=growing_transformer.device) * self.config.step_size
        )

        # create update direction for weight and bias
        self._value_weight = Parameter(
            torch.empty(self.heads * num_novel, self.d_model, device=growing_transformer.device)
        )
        self._value_bias = Parameter(torch.empty(self.heads * num_novel, device=growing_transformer.device))
        self._output_weight = Parameter(
            torch.empty(self.d_model, self.heads * num_novel, device=growing_transformer.device)
        )

        truncated_normal_(self._value_weight, mean=0.0, std=self.config.initializer_range)
        truncated_normal_(self._output_weight, mean=0.0, std=self.config.initializer_range)
        self._value_bias.data.zero_()

        return self.step_size.size()

    def degrow(self, selected: Tensor) -> None:
        with torch.no_grad():

            selected = selected.reshape(self.heads, -1)

            if selected.numel() == 0:
                self.reset_grow_state()
                return

            assert selected.size(0) == self.heads

            assert self.step_size is not None
            assert self._value_weight is not None
            assert self._value_bias is not None
            assert self._output_weight is not None

            d_new = selected.size(1)

            value_weight = torch.empty(self.heads, self.d_head + d_new, self.d_model, device=growing_transformer.device)
            value_bias = torch.empty(self.heads, self.d_head + d_new, device=growing_transformer.device)

            output_weight = torch.empty(
                self.d_model, self.heads, self.d_head + d_new, device=growing_transformer.device
            )

            # copy old neurons

            value_weight[:, : self.d_head] = self.value_linear.weight.view(self.heads, self.d_head, self.d_model)
            value_bias[:, : self.d_head] = self.value_linear.bias.view(self.heads, self.d_head)

            output_weight[..., : self.d_head] = self.output_linear.weight.view(self.d_model, self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.step_size.view(-1)[selected].view(self.heads, -1)

            value_weight[:, self.d_head :, :] = (
                self._value_weight.view(-1, self.d_model)[selected, :].view(self.heads, -1, self.d_model)
                * selected_steps[:, :, None]
            )
            value_bias[:, self.d_head :] = self._value_bias.view(-1)[selected].view(self.heads, -1) * selected_steps

            output_weight[..., self.d_head :] = self._output_weight.view(self.d_model, -1)[:, selected].view(
                self.d_model, self.heads, -1
            )

            self.d_head = self.d_head + d_new

            self.value_linear.weight = Parameter(value_weight.reshape(self.heads * self.d_head, self.d_model))
            self.value_linear.bias = Parameter(value_bias.reshape(self.heads * self.d_head))
            self.output_linear.weight = Parameter(output_weight.reshape(self.d_model, self.heads * self.d_head))

        self.reset_grow_state()

    def update_config(self, num_added: int):
        self.config.d_head_v = self.d_head + num_added


class ScaledDotProductAttention(GrowingModule):
    """Calculates attention matrix from query and key."""

    def __init__(self, config: GrowingConfig, heads: int = None):
        super().__init__(config=config)

        self.heads = first_not_none(heads, config.num_attention_heads)

        self.d_head = config.d_head_kq

        self.query_linear: Linear = Linear(config.d_model, self.heads * self.d_head)

        self.key_linear: Linear = Linear(config.d_model, self.heads * self.d_head)

        self.dropout = Dropout(config.attention_probs_dropout_prob)

        self.reset_grow_state()
        self.to(growing_transformer.device)

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.step_size = None

        # update directions (to be trained)
        self._weight: Optional[Parameter] = None
        self._bias: Optional[Parameter] = None

    def _direction_params(self) -> NamedDirectionParams:
        return {
            "direction_weight": self._weight,
            "direction_bias": self._bias,
        }

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

        if self.step_size is not None:
            assert self._weight is not None and self._bias is not None

            q_novel = torch.nn.functional.linear(x, self._weight[0], self._bias[0])
            q_novel = q_novel.view(batch_size, length, self.heads, -1)
            q_novel = q_novel * self.step_size

            k_novel = torch.nn.functional.linear(x, self._weight[1], self._bias[1])
            k_novel = k_novel.view(batch_size, length, self.heads, -1)

            product = product + torch.einsum(einsum_str, q_novel, k_novel)

        product = product / torch.sqrt(torch.tensor(self.d_head))

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in forward() function of model)
            product = product + attention_mask

        attention_probs = torch.softmax(product, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        return attention_probs

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        # add parameter to measure influence/gradient of adding new neurons
        self.step_size = Parameter(
            torch.ones(self.heads, num_novel, device=growing_transformer.device) * self.config.step_size,
            requires_grad=False,
        )

        # create update direction for weight and bias
        self._weight = Parameter(
            torch.empty(2, self.heads * num_novel, self.d_model, device=growing_transformer.device), requires_grad=False
        )

        self._bias = Parameter(
            torch.empty(2, self.heads * num_novel, device=growing_transformer.device), requires_grad=False
        )

        truncated_normal_(self._weight, mean=0.0, std=self.config.initializer_range)
        self._bias.data.zero_()

        return self.step_size.size()

    def degrow(self, selected: torch.Tensor) -> None:
        with torch.no_grad():

            if self.step_size is None:
                return

            assert self._weight is not None
            assert self._bias is not None

            selected = selected.reshape(self.heads, -1)

            d_new = selected.size(1)

            q_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features, device=growing_transformer.device)

            k_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features, device=growing_transformer.device)

            q_bias = torch.empty(self.heads, self.d_head + d_new, device=growing_transformer.device)

            k_bias = torch.empty(self.heads, self.d_head + d_new, device=growing_transformer.device)

            # copy old neurons

            q_weight[:, : self.d_head] = self.query_linear.weight.data.view(self.heads, self.d_head, self.d_model)
            k_weight[:, : self.d_head] = self.key_linear.weight.data.view(self.heads, self.d_head, self.d_model)
            q_bias[:, : self.d_head] = self.query_linear.bias.data.view(self.heads, self.d_head)
            k_bias[:, : self.d_head] = self.key_linear.bias.data.view(self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.step_size.view(-1)[selected].view(self.heads, -1)

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

            self.query_linear.weight = Parameter(q_weight.reshape(self.heads * self.d_head, self.d_model))
            self.key_linear.weight = Parameter(k_weight.reshape(self.heads * self.d_head, self.d_model))
            self.query_linear.bias = Parameter(q_bias.reshape(self.heads * self.d_head))
            self.key_linear.bias = Parameter(k_bias.reshape(self.heads * self.d_head))

        self.reset_grow_state()

    def update_config(self, num_added: int):
        self.config.d_head_kq = self.d_head + num_added


class GrowingAttention(GrowingModule):
    _bert_state_dict_map = {
        "self.query": "dot_product.query_linear",
        "self.key": "dot_product.key_linear",
        "self.value": "output.value_linear",
        "output.dense": "output.output_linear",
        "output.LayerNorm": "layer_norm",
    }

    def __init__(self, config):
        super().__init__(config=config)

        self.heads = config.num_attention_heads
        self.d_model = config.d_model

        self.dot_product = ScaledDotProductAttention(config=config)
        self.output = AttentionOutput(config=config)
        self.layer_norm = LayerNorm(self.d_model, eps=config.layer_norm_eps)

        self.dropout = Dropout(config.hidden_dropout_prob)

        self.reset_grow_state()
        self.to(growing_transformer.device)

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        influence_factor: float = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        attention = self.dot_product(x, attention_mask=attention_mask)

        out = self.output(x, attention=attention)

        if self.step_size is not None:
            assert self._new_dot_product is not None
            assert self._new_output is not None

            new_attention = self._new_dot_product(x, attention_mask=attention_mask)

            new_attention = self.step_size.reshape(1, -1, 1, 1) * new_attention

            out += self._new_output(x, new_attention)

        if influence_factor is not None:
            out = out * influence_factor

        out = self.dropout(out)
        out = self.layer_norm(out + x)

        if not return_attention:
            return out
        else:
            return out, attention

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept heads)
        self.step_size = None

        # update directions (to be trained) after growing
        self._new_dot_product: Optional[ScaledDotProductAttention] = None
        self._new_output: Optional[AttentionOutput] = None

    def _direction_params(self) -> NamedDirectionParams:
        params: Dict[str, Optional[Parameter]] = {}

        if self._new_dot_product is not None:
            for n, p in self._new_dot_product.named_parameters():
                params[f"dot_product_direction_{n}"] = p

        if self._new_output is not None:
            for n, p in self._new_output.named_parameters():
                params[f"output_direction_{n}"] = p

        return params

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        if num_novel > 0:
            step_size = self.config.step_size
            self.step_size = Parameter(
                torch.ones(num_novel, device=growing_transformer.device) * step_size,
            )

            self._new_dot_product = ScaledDotProductAttention(config=self.config, heads=num_novel)
            self._new_output = AttentionOutput(config=self.config, heads=num_novel, bias=False)

            truncated_normal_(self._new_output.output_linear.weight, mean=0.0, std=self.config.initializer_range)

            # set training flag of all new modules to own state
            self.train(self.training)

            return self.step_size.size()
        else:
            return torch.Size([0])

    def degrow(self, selected: torch.Tensor) -> None:
        # join old modules and selected heads of new modules together

        if self.step_size is None:
            return

        if selected.size(0) == 0:
            self.reset_grow_state()
            return

        assert self._new_dot_product is not None
        assert self._new_output is not None

        def concat_weights(a: Tensor, b: Tensor, out=False, mul_step=False):
            if out:
                a, b = a.T, b.T

            bias = len(a.size()) == 1

            if bias:
                a = a.reshape(self.heads, -1)
                b = b.reshape(-1, a.size(1))
            else:
                a = a.reshape(self.heads, -1, self.d_model)
                b = b.reshape(-1, a.size(1), self.d_model)

            b = b[selected]
            if mul_step:
                assert self.step_size is not None

                if bias:
                    step_size = self.step_size[selected, None]
                else:
                    step_size = self.step_size[selected, None, None]
                b = b * step_size

            ab = torch.cat([a, b], dim=0)

            if bias:
                ab = ab.reshape(-1)
            else:
                ab = ab.reshape(-1, self.d_model)

            if out:
                ab = ab.T

            return Parameter(ab)

        # adjust dot product
        self.dot_product.query_linear.weight = concat_weights(
            self.dot_product.query_linear.weight,
            self._new_dot_product.query_linear.weight,
        )

        self.dot_product.query_linear.bias = concat_weights(
            self.dot_product.query_linear.bias, self._new_dot_product.query_linear.bias
        )

        self.dot_product.query_linear.out_features = self.dot_product.query_linear.bias.size(0)

        self.dot_product.key_linear.weight = concat_weights(
            self.dot_product.key_linear.weight,
            self._new_dot_product.key_linear.weight,
        )

        self.dot_product.key_linear.bias = concat_weights(
            self.dot_product.key_linear.bias, self._new_dot_product.key_linear.bias
        )

        self.dot_product.key_linear.out_features = self.dot_product.key_linear.bias.size(0)

        # adjust output
        self.output.value_linear.weight = concat_weights(
            self.output.value_linear.weight, self._new_output.value_linear.weight, mul_step=True
        )

        self.output.value_linear.bias = concat_weights(
            self.output.value_linear.bias, self._new_output.value_linear.bias, mul_step=True
        )

        self.output.value_linear.out_features = self.output.value_linear.bias.size(0)

        self.output.output_linear.weight = concat_weights(
            self.output.output_linear.weight, self._new_output.output_linear.weight, out=True
        )

        self.heads = self.heads + selected.size(0)
        self.dot_product.heads = self.heads
        self.output.heads = self.heads

        self.reset_grow_state()

    def update_config(self, num_added: int):
        self.config.num_attention_heads = self.heads + num_added
