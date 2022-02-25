import torch
from torch.nn import functional as F
from contextlib import contextmanager
from typing import Optional


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=torch.nn.Tanh()):
        super().__init__()

        self.linear_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.linear_out = torch.nn.Linear(hidden_features, out_features)
        self.hidden_features = hidden_features

        self.reset_grow_state()

    def reset_grow_state(self):
        # update directions (to be trained)
        self.weight_dir = None
        self.bias_dir = None

        # step size (used to calculate gradients for selecting kept neurons)
        self.new_neurons = None

        self.was_split = False

    @property
    def in_features(self):
        return self.linear_in.in_features

    @property
    def out_features(self):
        return self.linear_out.out_features

    @property
    def num_new_neurons(self):
        """Return total number of neurons that were added."""
        if self.new_neurons is None:
            return 0
        else:
            return self.new_neurons.size(0)

    @contextmanager
    def some_grad_only(self, *some_parameters):
        # temporarily save requires_grad for all parameters
        _requires_grad = [p.requires_grad for p in self.parameters()]

        # disable all grads
        for p in self.parameters():
            p.requires_grad = False

        # enable grads some parameters
        for p in some_parameters:
            if p is not None:
                p.requires_grad = True

        yield  # yield for forward pass

        # reset requires_grad of all parameters
        for p, rg in zip(self.parameters(), _requires_grad):
            p.requires_grad = rg

    @contextmanager
    def new_grad_only(self):
        with self.some_grad_only(self.new_neurons):
            yield

    @contextmanager
    def direction_grad_only(self):
        with self.some_grad_only(*self.direction_params()):
            yield

    def direction_params(self):
        return [
            self.weight_dir,
            self.bias_dir
        ]

    def forward(self, x):
        h = self.linear_in(x)

        if self.new_neurons is not None and self.was_split:
            w_noise = F.linear(x,
                self.weight_dir[:self.hidden_features],
                self.bias_dir[:self.hidden_features]
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
                h_new = F.linear(x,
                    self.weight_dir[-num_novel:],
                    self.bias_dir[-num_novel:]
                ) * self.new_neurons[-num_novel:]

                y_add = self.activation(h_new)

                # equivalent to using a weight of one for each output neuron
                y_add = y_add.sum(-1, keepdim=True)

                y = y + y_add

        return y

    def grow(self,
              split : bool = True,
              num_novel : int = 0,
              step_size = 1e-2,
              eps : float = 1e-1,
              eps_split : Optional[float] = None,
              eps_novel : Optional[float] = None,
              eps_split_weight : Optional[float] = None,
              eps_split_bias : Optional[float] = None,
              eps_novel_weight : Optional[float] = None,
              eps_novel_bias : Optional[float] = None):

        self.was_split = split

        num_new = self.hidden_features * split + num_novel

        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = torch.nn.Parameter(
            torch.ones(num_new) * step_size,
            requires_grad=False
        )

        # create update direction for weight and bias
        self.weight_dir = torch.nn.Parameter(torch.empty(
            num_new, self.in_features
        ), requires_grad=False)

        self.bias_dir = torch.nn.Parameter(torch.empty(num_new), requires_grad=False)

        # initialize directions
        if split:
            e = eps_split_weight or eps_split or eps
            torch.nn.init.uniform_(
                self.weight_dir[:self.hidden_features],
                -e, e)

            e = eps_split_bias or eps_split or eps
            torch.nn.init.uniform_(
                self.bias_dir[:self.hidden_features],
                -e, e)

        if num_novel > 0:
            e = eps_novel_weight or eps_novel or eps
            torch.nn.init.uniform_(
                self.weight_dir[-num_novel:],
                -e, e)

            e = eps_novel_bias or eps_novel or eps
            torch.nn.init.uniform_(
                self.bias_dir[-num_novel],
                -e, e)

        return num_new

    def select(self, k):
        assert self.new_neurons is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(torch.abs(self.new_neurons.grad), k).indices

    def degrow(self, selected : torch.Tensor):
        with torch.no_grad():
            num_old = self.hidden_features

            # split neurons to keep
            split = selected[selected < num_old * self.was_split]
            num_split = split.size(0)
            # novel neurons to add
            novel = selected[selected >= num_old * self.was_split]

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

            if split.size(0) > 0:
                weight_noise =  self.weight_dir[split] * self.new_neurons[split, None]

                weight_in[split] = self.linear_in.weight[split] + weight_noise
                weight_in[num_old:num_old + num_split] = self.linear_in.weight[split] - weight_noise

                bias_noise =  self.bias_dir[split] * self.new_neurons[split]

                bias_in[split] = self.linear_in.bias[split] + bias_noise
                bias_in[num_old:num_old + num_split] = self.linear_in.bias[split] - bias_noise

                # for output layer, copy half the weights
                weight_out[:, split] = self.linear_out.weight[:, split] * 0.5
                weight_out[:, num_old:num_old + num_split] = self.linear_out.weight[:, split] * 0.5

            if novel.size(0) > 0:
                if not self.was_split:
                    novel -= num_old

                # copy new neurons
                weight_in[num_old + num_split:] = self.weight_dir[novel] * self.new_neurons[novel, None]
                bias_in[num_old + num_split:] = self.bias_dir[novel] * self.new_neurons[novel]

                # for output layer initialize 1
                weight_out[num_old + num_split:] = 1.0


            self.linear_in.weight = torch.nn.Parameter(weight_in)
            self.linear_in.bias = torch.nn.Parameter(bias_in)
            self.linear_out.weight = torch.nn.Parameter(weight_out)

        # adjust features
        self.hidden_features = self.linear_in.weight.size(0)

        self.reset_grow_state()
