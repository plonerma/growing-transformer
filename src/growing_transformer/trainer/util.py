from itertools import product
from typing import Any, Iterator, Mapping


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
