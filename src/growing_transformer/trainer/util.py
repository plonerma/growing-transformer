import random
from itertools import product
from typing import Any, Iterator, Mapping

from torch.utils.data.dataset import Subset


def log_line(log):
    log.info("-" * 100)


class GridSearch:
    def __init__(self, grid: Mapping[str, Any]):
        self.grid = grid

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        keys, values = zip(*self.grid.items())
        return (dict(zip(keys, p)) for p in product(*values))

    def __len__(self):
        _len = 1
        for v in self.grid.values():
            _len *= len(v)
        return _len


def subsample(data, proportion=0.1):
    """Subsample dataset."""
    n_samples = len(data)
    indices = list(range(n_samples))
    random.shuffle(indices)

    # calculate new number of samples
    n_samples = round(proportion * n_samples)

    # create new dataset
    subset = indices[:n_samples]
    return Subset(data, subset)
