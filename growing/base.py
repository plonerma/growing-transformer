import torch

from typing import List, Mapping, Any, Iterable, Optional
from contextlib import contextmanager
from abc import abstractmethod

class GrowingModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.new_neurons: Optional[torch.nn.Parameter] = None
        self.config = config

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

    @property
    def num_new_neurons(self) -> int:
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

    def grow(self, step_size: float = 1e-1) -> List[Any]:
        sizes = [self._grow(step_size)]

        for child in self.growing_children():
            sizes.append(child.grow(step_size))

        return sizes

    def degrow(self, selected: List[Any]) -> None:
        self._degrow(selected[0])

        for s, child in zip(selected[1:], self.growing_children()):
            child.degrow(s)

    @abstractmethod
    def _degrow(self, selected: torch.Tensor) -> None:
        pass

    @abstractmethod
    def _grow(self, step_size: float = 1e-1) -> torch.Size:
        pass

    @abstractmethod
    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        pass

    def direction_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self._direction_params() if p is not None]

    def select(self, k: int) -> torch.Tensor:
        assert self.new_neurons is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(torch.abs(self.new_neurons.grad), k).indices

    def get_config(self, *args: str, default: Any = None):
        for arg in args:
            try:
                return self.config[arg]
            except KeyError:
                pass
        return default
