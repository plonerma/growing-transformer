from abc import abstractmethod
from typing import Any, Iterable, List, Mapping, Optional

import torch


class Growing(torch.nn.Module):
    def __init__(self, config: Mapping[str, Any] = {}):
        super().__init__()
        self.config = config

    def growing_children(self):
        for child in self.children():
            if isinstance(child, Growing):
                yield child

    def growing_modules(self):
        for m in self.modules():
            if isinstance(m, Growing):
                yield m

    def direction_params(self) -> Iterable[torch.nn.Parameter]:
        for m in self.growing_modules():
            if not isinstance(m, GrowingModule):
                continue

            for p in m._direction_params():
                if p is not None:
                    yield p

    def new_params(self) -> Iterable[torch.nn.Parameter]:
        for m in self.growing_modules():
            if not isinstance(m, GrowingModule):
                continue

            p = m.new_neurons

            if p is not None:
                yield p

    def get_config(self, *args: str, default: Any = None):
        for arg in args:
            try:
                return self.config[arg]
            except KeyError:
                pass
        return default

    def degrow(self, selected: torch.Tensor) -> None:
        pass

    def grow(self) -> torch.Size:
        return torch.Size()

    def select(self, k: int) -> torch.Tensor:
        return torch.Tensor()


class GrowingModule(Growing):
    def __init__(self, config: Mapping[str, Any] = {}):
        super().__init__(config)
        self.new_neurons: Optional[torch.nn.Parameter] = None

    @property
    def num_new_neurons(self) -> int:
        """Return total number of neurons that were added."""
        if self.new_neurons is None:
            return 0
        else:
            return self.new_neurons.size(0)

    @abstractmethod
    def degrow(self, selected: torch.Tensor) -> None:
        pass

    @abstractmethod
    def grow(self) -> torch.Size:
        pass

    @abstractmethod
    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        pass

    def select(self, k: int) -> torch.Tensor:
        assert self.new_neurons is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(self.new_neurons * self.new_neurons.grad, k).indices
