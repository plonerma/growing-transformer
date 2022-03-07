import torch
from torch.nn import Parameter
from torch.nn.init import uniform_

from contextlib import contextmanager
from typing import Optional, List, Mapping, Any

from . import GrowingModule


class MLP(GrowingModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 activation=torch.nn.Tanh(),
                 config: Mapping[str, Any] = {}):
        super().__init__(config)

        self.linear_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.linear_out = torch.nn.Linear(hidden_features, out_features)
        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.new_neurons: Optional[torch.nn.Parameter] = None

        # update directions (to be trained)
        self._in_weight_split: Optional[torch.nn.Parameter] = None
        self._in_bias_split: Optional[torch.nn.Parameter] = None
        self._out_weight_novel: Optional[torch.nn.Parameter] = None
        self._in_weight_novel: Optional[torch.nn.Parameter] = None
        self._in_bias_novel: Optional[torch.nn.Parameter] = None

    def _direction_params(self) -> List[Optional[torch.nn.Parameter]]:
        return [
            self._in_weight_split,
            self._in_bias_split,
            self._out_weight_novel,
            self._in_weight_novel,
            self._in_bias_novel,
        ]

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

        was_split = self.new_neurons is not None and self._in_weight_split is not None

        if was_split:
            assert self.new_neurons is not None
            assert self._in_weight_split is not None
            assert self._in_bias_split is not None

            w_noise = torch.nn.functional.linear(x,
                self._in_weight_split,
                self._in_bias_split
            ) * self.new_neurons[:self.hidden_features]

            h_plus = self.activation(h + w_noise)
            h_minus = self.activation(h - w_noise)

            h = 0.5 * (h_plus + h_minus)
        else:
            h = self.activation(h)

        y = self.linear_out(h)

        if self.new_neurons is not None:
            num_novel = self.num_new_neurons - was_split * self.hidden_features

            if num_novel > 0:
                assert self._in_weight_novel is not None
                assert self._in_bias_novel is not None
                assert self._out_weight_novel is not None

                h_novel = torch.nn.functional.linear(x, self._in_weight_novel, self._in_bias_novel)
                y_novel = self.activation(h_novel) * self.new_neurons[-num_novel:]
                y_novel = torch.nn.functional.linear(y_novel, self._out_weight_novel)
                y = y + y_novel

        return y

    def grow(self) -> torch.Size:
        step_size = self.get_config('step_size', default=1e-2)
        split = self.get_config('split', default=True)
        num_novel = self.get_config('num_novel', default=0)
        eps_split_weight = self.get_config('eps_split_weight', 'eps_split', default=1e-1)
        eps_split_bias = self.get_config('eps_split_bias', 'eps_split', default=1e-1)
        eps_novel_weight = self.get_config('eps_novel_weight', 'eps_novel', default=1e-1)
        eps_novel_bias =self.get_config('eps_novel_bias', 'eps_novel', default=1e-1)

        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = Parameter(torch.ones(self.hidden_features * split + num_novel) * step_size,requires_grad=False)

        # create update direction for weight and bias
        if split:
            self._in_weight_split = Parameter(torch.empty(self.hidden_features, self.in_features), requires_grad=False)
            self._in_bias_split = Parameter(torch.empty(self.hidden_features), requires_grad=False)
            uniform_(self._in_weight_split,-eps_split_weight, eps_split_weight)
            uniform_(self._in_bias_split,-eps_split_bias, eps_split_bias)

        if num_novel > 0:
            self._in_weight_novel = Parameter(torch.empty(num_novel, self.in_features), requires_grad=False)
            self._out_weight_novel = Parameter(torch.empty(self.out_features, num_novel), requires_grad=False)
            self._in_bias_novel = Parameter(torch.empty(num_novel), requires_grad=False)
            uniform_(self._in_weight_novel,-eps_novel_weight, eps_novel_weight)
            uniform_(self._out_weight_novel,-eps_novel_weight, eps_novel_weight)
            uniform_(self._in_bias_novel,-eps_novel_bias, eps_novel_bias)

        return self.new_neurons.size()

    def degrow(self, selected: torch.Tensor) -> None:
        with torch.no_grad():
            assert self.new_neurons is not None

            num_old = self.hidden_features

            was_split = self._in_weight_split is not None

            if was_split:
                # split neurons to keep
                split = selected[selected < num_old]
                num_split = split.size(0)
                # novel neurons to add
                novel = selected[selected >= num_old] - self.hidden_features
            else:
                num_split = 0
                novel = selected

            weight_in = torch.empty(
                num_old + selected.size(0),
                self.in_features
            )

            weight_out = torch.empty(
                self.out_features,
                num_old + selected.size(0),
            )

            bias_in = torch.empty(num_old + selected.size(0))

            # copy old neurons (split neurons will be overwritten)
            weight_in[:num_old] = self.linear_in.weight
            bias_in[:num_old] = self.linear_in.bias
            weight_out[:,:num_old] = self.linear_out.weight

            # copy split neurons (with update direction)

            if num_split > 0:
                assert self._in_weight_split is not None
                assert self._in_bias_split is not None
                assert self.new_neurons is not None

                weight_noise =  self._in_weight_split[split] * self.new_neurons[split, None]

                weight_in[split] = self.linear_in.weight[split] + weight_noise
                weight_in[num_old:num_old + num_split] = self.linear_in.weight[split] - weight_noise

                bias_noise =  self._in_bias_split[split] * self.new_neurons[split]

                bias_in[split] = self.linear_in.bias[split] + bias_noise
                bias_in[num_old:num_old + num_split] = self.linear_in.bias[split] - bias_noise

                # for output layer, copy half the weights
                weight_out[:, split] = self.linear_out.weight[:, split] * 0.5
                weight_out[:, num_old:num_old + num_split] = self.linear_out.weight[:, split] * 0.5

            if novel.size(0) > 0:
                assert self._in_weight_novel is not None
                assert self._in_bias_novel is not None
                assert self._out_weight_novel is not None

                # copy new neurons
                weight_in[num_old + num_split:] = self._in_weight_novel[novel]
                bias_in[num_old + num_split:] = self._in_bias_novel[novel]
                weight_out[:, num_old + num_split:] = self._out_weight_novel[:, novel] * self.new_neurons[None, novel]


            self.linear_in.weight = Parameter(weight_in)
            self.linear_in.bias = Parameter(bias_in)
            self.linear_out.weight = Parameter(weight_out)

        # adjust features
        self.linear_in.out_features, self.linear_in.in_features = self.linear_in.weight.size()
        self.linear_out.out_features, self.linear_out.in_features = self.linear_out.weight.size()
        self.reset_grow_state()
