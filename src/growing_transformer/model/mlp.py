from typing import Callable, Optional

import torch
from torch.nn import Parameter
from transformers.activations import ACT2FN

import growing_transformer

from ..configuration import GrowingConfig
from .base import GrowingModule, NamedDirectionParams, truncated_normal_


class GrowingMLP(GrowingModule):
    def __init__(
        self,
        config: GrowingConfig,
        *,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__(config=config)

        in_features = in_features or config.d_model
        hidden_features = hidden_features or config.intermediate_size
        out_features = out_features or config.d_model

        self.linear_in = torch.nn.Linear(in_features, hidden_features)
        self.linear_out = torch.nn.Linear(hidden_features, out_features)

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        self.activation: Callable
        if isinstance(self.config.hidden_act, str):
            self.activation = ACT2FN[self.config.hidden_act]
        else:
            self.activation = self.config.hidden_act

        self.reset_grow_state()
        self.to(growing_transformer.device)

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.step_size: Optional[torch.nn.Parameter] = None

        # update directions (to be trained)
        self._in_split_weight: Optional[torch.nn.Parameter] = None
        self._in_split_bias: Optional[torch.nn.Parameter] = None
        self._out_novel_weight: Optional[torch.nn.Parameter] = None
        self._in_novel_weight: Optional[torch.nn.Parameter] = None
        self._in_novel_bias: Optional[torch.nn.Parameter] = None

    def _direction_params(self, named=False) -> NamedDirectionParams:
        return {
            "split_direction_weight": self._in_split_weight,
            "split_direction_bias": self._in_split_bias,
            "novel_direction_out_weight": self._out_novel_weight,
            "novel_direction_in_weight": self._in_novel_weight,
            "novel_direction_in_bias": self._in_novel_bias,
        }

    @property
    def in_features(self) -> int:
        return self.linear_in.in_features

    @property
    def hidden_features(self) -> int:
        return self.linear_in.out_features

    @property
    def out_features(self) -> int:
        return self.linear_out.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear_in(x)

        was_split = self.step_size is not None and self._in_split_weight is not None

        if was_split:
            assert self.step_size is not None
            assert self._in_split_weight is not None
            assert self._in_split_bias is not None

            w_noise = (
                torch.nn.functional.linear(x, self._in_split_weight, self._in_split_bias)
                * self.step_size[: self.hidden_features]
            )

            h_plus = self.activation(h + w_noise)
            h_minus = self.activation(h - w_noise)

            h = 0.5 * (h_plus + h_minus)
        else:
            h = self.activation(h)

        y = self.linear_out(h)

        if self.step_size is not None:
            num_novel = self.num_step_size - was_split * self.hidden_features

            if num_novel > 0:
                assert self._in_novel_weight is not None
                assert self._in_novel_bias is not None
                assert self._out_novel_weight is not None

                h_novel = torch.nn.functional.linear(x, self._in_novel_weight, self._in_novel_bias)
                y_novel = self.activation(h_novel) * self.step_size[-num_novel:]
                y_novel = torch.nn.functional.linear(y_novel, self._out_novel_weight)
                y = y + y_novel

        y = self.dropout(y)

        return y

    def grow(self, num_novel: int = 0, split: bool = True) -> torch.Size:
        # add parameter to measure influence/gradient of adding new neurons
        self.step_size = Parameter(
            torch.ones(self.hidden_features * split + num_novel, device=growing_transformer.device)
            * self.config.step_size
        )

        # create update direction for weight and bias
        if split:
            self._in_split_weight = Parameter(
                torch.empty(self.hidden_features, self.in_features, device=growing_transformer.device),
            )
            self._in_split_bias = Parameter(torch.empty(self.hidden_features, device=growing_transformer.device))

            truncated_normal_(self._in_split_weight, mean=0.0, std=self.config.init_split_range)
            truncated_normal_(self._in_split_bias, mean=0.0, std=self.config.init_split_range)

        if num_novel > 0:
            self._in_novel_weight = Parameter(
                torch.empty(num_novel, self.in_features, device=growing_transformer.device)
            )
            self._out_novel_weight = Parameter(
                torch.empty(self.out_features, num_novel, device=growing_transformer.device)
            )
            self._in_novel_bias = Parameter(torch.empty(num_novel, device=growing_transformer.device))

            truncated_normal_(self._in_novel_weight, mean=0.0, std=self.config.initializer_range)
            truncated_normal_(self._out_novel_weight, mean=0.0, std=self.config.initializer_range)
            self._in_novel_bias.data.zero_()

        return self.step_size.size()

    def degrow(self, selected: torch.Tensor) -> None:
        with torch.no_grad():
            assert self.step_size is not None

            num_old = self.hidden_features

            was_split = self._in_split_weight is not None

            if was_split:
                # split neurons to keep
                split = selected[selected < num_old]
                num_split = split.size(0)
                # novel neurons to add
                novel = selected[selected >= num_old] - self.hidden_features
            else:
                num_split = 0
                novel = selected

            weight_in = torch.empty(num_old + selected.size(0), self.in_features, device=growing_transformer.device)

            weight_out = torch.empty(self.out_features, num_old + selected.size(0), device=growing_transformer.device)

            bias_in = torch.empty(num_old + selected.size(0), device=growing_transformer.device)

            # copy old neurons (split neurons will be overwritten)
            weight_in[:num_old] = self.linear_in.weight
            bias_in[:num_old] = self.linear_in.bias
            weight_out[:, :num_old] = self.linear_out.weight

            # copy split neurons (with update direction)

            if num_split > 0:
                assert self._in_split_weight is not None
                assert self._in_split_bias is not None
                assert self.step_size is not None

                weight_noise = self._in_split_weight[split] * self.step_size[split, None]

                weight_in[split] = self.linear_in.weight[split] + weight_noise
                weight_in[num_old : num_old + num_split] = self.linear_in.weight[split] - weight_noise

                bias_noise = self._in_split_bias[split] * self.step_size[split]

                bias_in[split] = self.linear_in.bias[split] + bias_noise
                bias_in[num_old : num_old + num_split] = self.linear_in.bias[split] - bias_noise

                # for output layer, copy half the weights
                weight_out[:, split] = self.linear_out.weight[:, split] * 0.5
                weight_out[:, num_old : num_old + num_split] = self.linear_out.weight[:, split] * 0.5

            if novel.size(0) > 0:
                assert self._in_novel_weight is not None
                assert self._in_novel_bias is not None
                assert self._out_novel_weight is not None

                # copy new neurons
                weight_in[num_old + num_split :] = self._in_novel_weight[novel]
                bias_in[num_old + num_split :] = self._in_novel_bias[novel]
                weight_out[:, num_old + num_split :] = (
                    self._out_novel_weight[:, novel] * self.step_size[None, novel + self.hidden_features * was_split]
                )

            self.linear_in.weight = Parameter(weight_in)
            self.linear_in.bias = Parameter(bias_in)
            self.linear_out.weight = Parameter(weight_out)

        # adjust features
        self.linear_in.out_features, self.linear_in.in_features = self.linear_in.weight.size()
        self.linear_out.out_features, self.linear_out.in_features = self.linear_out.weight.size()

        # reset temporary variables
        self.reset_grow_state()

    def update_config(self, num_added: int):
        self.config.intermediate_size = self.linear_in.out_features + num_added
