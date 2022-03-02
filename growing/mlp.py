import torch

from contextlib import contextmanager
from typing import Optional

from .base import GrowingModule


class MLP(GrowingModule):
    def __init__(self, in_features, out_features, hidden_features, activation=torch.nn.Tanh()):
        super().__init__()

        self.linear_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.linear_out = torch.nn.Linear(hidden_features, out_features)

    @property
    def in_features(self):
        return self.linear_in.in_features

    @property
    def hidden_features(self):
        return self.linear_in.out_features

    @property
    def out_features(self):
        return self.linear_out.out_features

    def forward(self, x):
        h = self.linear_in(x)

        if self.new_neurons is not None and self.was_split:
            w_noise = torch.nn.functional.linear(x,
                self._weight_dir[:self.hidden_features],
                self._bias_dir[:self.hidden_features]
            ) * self.new_neurons[:self.hidden_features]

            h_plus = self.activation(h + w_noise)
            h_minus = self.activation(h - w_noise)

            h = 0.5 * (h_plus + h_minus)
        else:
            h = self.activation(h)

        y = self.linear_out(h)

        if self.new_neurons is not None:
            num_novel = self.num_new_neurons - self.was_split * self.hidden_features

            if num_novel > 0:
                h_novel = torch.nn.functional.linear(x,
                    self._weight_dir[-num_novel:],
                    self._bias_dir[-num_novel:]
                )

                y_novel = self.activation(h_novel) * self.new_neurons[-num_novel:]

                y_novel = y_novel.sum(-1, keepdim=True)

                y = y + y_novel

        return y

    def grow(self,
             split : bool = True,
             num_novel : int = 0,
             step_size = 1,
             eps_split : float = 1e-1,
             eps_novel : float = 1e-2,
             eps_split_weight : Optional[float] = None,
             eps_split_bias : Optional[float] = None,
             eps_novel_weight : Optional[float] = None,
             eps_novel_bias : Optional[float] = None,
             **kw):

        self.was_split = split

        num_new = self.hidden_features * split + num_novel

        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = torch.nn.Parameter(
            torch.ones(num_new) * step_size,
            requires_grad=False
        )

        # create update direction for weight and bias
        self._weight_dir = torch.nn.Parameter(torch.empty(
            num_new, self.in_features
        ), requires_grad=False)

        self._bias_dir = torch.nn.Parameter(torch.empty(num_new), requires_grad=False)

        # initialize directions
        if split:
            e = eps_split_weight or eps_split
            torch.nn.init.uniform_(
                self._weight_dir[:self.hidden_features],
                -e, e)

            e = eps_split_bias or eps_split
            torch.nn.init.uniform_(
                self._bias_dir[:self.hidden_features],
                -e, e)

        if num_novel > 0:
            e = eps_novel_weight or eps_novel
            torch.nn.init.uniform_(
                self._weight_dir[-num_novel:],
                -e, e)

            e = eps_novel_bias or eps_novel
            torch.nn.init.uniform_(
                self._bias_dir[-num_novel:],
                -e, e)
        return self.new_neurons.size()

    def degrow(self, selected : torch.Tensor):
        with torch.no_grad():
            num_old = self.hidden_features


            if self.was_split:
                # split neurons to keep
                split = selected[selected < num_old * self.was_split]
                num_split = split.size(0)
                # novel neurons to add
                novel = selected[selected >= num_old * self.was_split]
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
            weight_out[:, :num_old] = self.linear_out.weight

            # copy split neurons (with update direction)

            if num_split > 0:
                weight_noise =  self._weight_dir[split] * self.new_neurons[split, None]

                weight_in[split] = self.linear_in.weight[split] + weight_noise
                weight_in[num_old:num_old + num_split] = self.linear_in.weight[split] - weight_noise

                bias_noise =  self._bias_dir[split] * self.new_neurons[split]

                bias_in[split] = self.linear_in.bias[split] + bias_noise
                bias_in[num_old:num_old + num_split] = self.linear_in.bias[split] - bias_noise

                # for output layer, copy half the weights
                weight_out[:, split] = self.linear_out.weight[:, split] * 0.5
                weight_out[:, num_old:num_old + num_split] = self.linear_out.weight[:, split] * 0.5

            if novel.size(0) > 0:

                # copy new neurons
                weight_in[num_old + num_split:] = self._weight_dir[novel]
                bias_in[num_old + num_split:] = self._bias_dir[novel]

                # for output layer initialize with step size
                weight_out[:, num_old + num_split:] = self.new_neurons[None, novel]


            self.linear_in.weight = torch.nn.Parameter(weight_in)
            self.linear_in.bias = torch.nn.Parameter(bias_in)
            self.linear_out.weight = torch.nn.Parameter(weight_out)

        # adjust features
        self.linear_in.out_features, self.linear_in.in_features = self.linear_in.weight.size()
        self.linear_out.out_features, self.linear_out.in_features = self.linear_out.weight.size()
        self.reset_grow_state()
