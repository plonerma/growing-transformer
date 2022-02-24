import torch

from contextlib import contextmanager

from .base import Linear

class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=torch.nn.Sigmoid()):
        super().__init__()

        self.linear_in = Linear(in_features, hidden_features)
        self.activation = activation
        self.linear_out = Linear(hidden_features, out_features)

        self.reset_grow_state()

    def reset_grow_state(self):
        self._new_neurons = None
        self._was_split = None

    @property
    def in_features(self):
        return self.linear_in.in_features

    @property
    def out_features(self):
        return self.linear_out.out_features

    @property
    def hidden_features(self):
        return self.linear_in.out_features

    @property
    def num_new_neurons(self):
        """Return total number of neurons that were added."""
        if self._new_neurons is None:
            return 0
        else:
            return self._new_neurons.size(0)

    @contextmanager
    def some_grad_only(self, *some_parameters):
        assert self._new_neurons is not None

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
        with self.some_grad_only(self._new_neurons):
            yield

    @contextmanager
    def direction_grad_only(self):
        with self.some_grad_only(*self.direction_params()):
            yield

    def direction_params(self):
        return [p for p in (
            self.linear_in.bias_dir,
            self.linear_in.weight_dir,
            self.linear_out.bias_dir,
            self.linear_out.weight_dir,
        ) if p is not None]

    def forward(self, x):
        x = self.linear_in(x)

        if self._new_neurons is not None:
            old_h = self.hidden_features - self.num_new_neurons

            if self._was_split:
                x = x * torch.concat([
                    self._new_neurons[:old_h],  # old neuons
                    self._new_neurons[:old_h],  # copy of old neurons
                ] + (
                    [self._new_neurons[old_h:]]  # novel neurons
                    if self.num_new_neurons > old_h
                    else []
                ))
            else:
                if self.num_new_neurons > 0:
                    x[..., -self.num_new_neurons:] = x[..., -self.num_new_neurons:]  * self._new_neurons

        x = self.activation(x)

        x = self.linear_out(x)
        return x

    def grow(self, **kw):
        self._was_split = kw.get('split', True)

        n = kw.get('num_novel', 0)
        if self._was_split:
            n += self.hidden_features

        # add parameter to measure influence/gradient of adding new neurons
        self._new_neurons = torch.nn.Parameter(torch.ones(n), requires_grad=False)

        # grow both linear layers
        self.linear_out.grow(dim=1, **kw)
        self.linear_in.grow(dim=0, **kw)

        return self.num_new_neurons

    def select(self, k):
        assert self._new_neurons is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(torch.abs(self._new_neurons.grad), k).indices

    def degrow(self, selected):
        kw = dict(
            selected=selected,
            split=self._was_split,
            num_old=self.hidden_features - self.num_new_neurons
        )

        self.linear_out.degrow(dim=1, **kw)
        self.linear_in.degrow(dim=0, **kw)
        self.reset_grow_state()
