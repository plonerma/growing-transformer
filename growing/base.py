import torch

from typing import List, Mapping, Any
from contextlib import contextmanager
from abc import abstractmethod

class GrowingModule(torch.nn.Module):
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

    def direction_params(self):
        return [
            self._weight_dir,
            self._bias_dir
        ]

    @property
    def num_new_neurons(self):
        """Return total number of neurons that were added."""
        if self.new_neurons is None:
            return 0
        else:
            return self.new_neurons.size(0)

    @contextmanager
    def new_grad_only(self):
        with self.some_grad_only(self.new_neurons):
            yield

    @contextmanager
    def direction_grad_only(self):
        with self.some_grad_only(*self.direction_params()):
            yield

    def growing_children(self):
        for child in self.children():
            if isinstance(child, GrowingModule):
                yield child

    def grow(self, kws : List[Mapping]) -> List[Any]:
        sizes = [self._grow(**kws[0])]

        for kw, child in zip(kws[1:], self.growing_children()):
            sizes.append(child.grow(kw))

        return sizes

    def degrow(self, selected : List[Any]):
        self._degrow(selected[0])

        for s, child in zip(selected[1:], self.growing_children()):
            child.degrow(s)

    @abstractmethod
    def _degrow(self, selected : torch.Tensor):
        pass

    @abstractmethod
    def _grow(self,
              split : bool = True,
              num_novel : int = 0,
              step_size = 1e-2,
              **kwargs) -> torch.Size:
        pass

    def select(self, k):
        assert self.new_neurons is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(torch.abs(self.new_neurons.grad), k).indices
