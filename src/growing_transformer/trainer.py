from contextlib import contextmanager

from . import Growing


class Trainer:
    def __init__(self, model):
        self.model = model

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
    def direction_grad_only(self):
        with self.some_grad_only(*model.direction_params()):
            yield

    @contextmanager
    def new_grad_only(self):
        with self.some_grad_only(*model.direction_params()):
            yield
