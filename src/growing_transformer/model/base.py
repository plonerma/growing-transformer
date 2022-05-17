from abc import abstractmethod
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import torch

from ..configuration import GrowingConfig

NamedDirectionParams = Dict[str, Optional[torch.nn.Parameter]]


def truncated_normal_(tensor, mean=0, std=1):
    """Source: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16"""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


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

    def _direction_params(self) -> NamedDirectionParams:
        return {}

    def direction_params(
        self, named=False, recursive=False
    ) -> Union[Iterable[torch.nn.Parameter], Iterable[Tuple[str, torch.nn.Parameter]]]:
        if recursive:
            for m in self.growing_modules():
                yield from m.direction_params(named=named, recursive=False)
        else:
            for p, n in self._direction_params().items():
                if p is not None:
                    if named:
                        yield p, n
                    else:
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

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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
    def _direction_params(self) -> NamedDirectionParams:
        pass

    def select(self, k: int) -> torch.Tensor:
        assert self.step_size is not None

        # return indices of neurons with largest absolute gradient
        return torch.topk(self.step_size * self.step_size.grad, k).indices

    @abstractmethod
    def update_config(self, num_added: int):
        pass
