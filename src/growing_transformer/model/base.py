from abc import abstractmethod
from typing import Iterable, Mapping, Optional

import torch

from ..configuration import GrowingConfig


class Growing(torch.nn.Module):
    _bert_state_dict_map: Optional[Mapping[str, str]] = None

    def __init__(self, config: GrowingConfig):
        super().__init__()
        self.config = config

        if config.bert_like_state_dict and self._bert_state_dict_map is not None:
            self._register_state_dict_hook(self._bert_state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_bert_state_dict_pre_hook)

    def growing_children(self):
        for child in self.children():
            if isinstance(child, Growing):
                yield child

    def growing_modules(self, named=False):
        for n, m in self.named_modules():
            if isinstance(m, Growing):
                if named:
                    yield n, m
                else:
                    yield m

    def direction_params(self) -> Iterable[torch.nn.Parameter]:
        for m in self.growing_modules():
            if not isinstance(m, GrowingModule):
                continue

            for p in m._direction_params():
                if p is not None:
                    yield p

    def step_size_params(self) -> Iterable[torch.nn.Parameter]:
        for m in self.growing_modules():
            if not isinstance(m, GrowingModule):
                continue

            p = m.step_size

            if p is not None:
                yield p

    def degrow(self, selected: torch.Tensor) -> None:
        pass

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        return torch.Size()

    def select(self, k: int) -> torch.Tensor:
        return torch.Tensor()

    def _switch_key_prefix(self, state_dict, old_prefix, new_prefix):
        for k in list(state_dict.keys()):
            if k.startswith(old_prefix):
                k_new = new_prefix + k[len(old_prefix) :]
                state_dict[k_new] = state_dict[k]
                del state_dict[k]

    @staticmethod
    def _bert_state_dict_hook(self, state_dict, prefix, local_metadata):
        for k_new_prefix, k_prefix in self._bert_state_dict_map.items():
            self._switch_key_prefix(state_dict, old_prefix=prefix + k_prefix, new_prefix=prefix + k_new_prefix)

    def _load_bert_state_dict_pre_hook(self, state_dict, prefix, *_):
        for k_prefix, k_new_prefix in self._bert_state_dict_map.items():
            self._switch_key_prefix(state_dict, old_prefix=prefix + k_prefix, new_prefix=prefix + k_new_prefix)

    def update_config(self, num_added: int):
        pass


class GrowingModule(Growing):
    def __init__(self, config: GrowingConfig):
        super().__init__(config)
        self.step_size: Optional[torch.nn.Parameter] = None

    @property
    def num_step_size(self) -> int:
        """Return total number of neurons that were added."""
        if self.step_size is None:
            return 0
        else:
            return self.step_size.size(0)

    @abstractmethod
    def degrow(self, selected: torch.Tensor) -> None:
        pass

    @abstractmethod
    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        pass

    @abstractmethod
    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        pass

    def select(self, k: int) -> torch.Tensor:
        assert self.step_size is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(self.step_size * self.step_size.grad, k).indices

    @abstractmethod
    def update_config(self, num_added: int):
        pass
