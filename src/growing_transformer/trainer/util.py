import logging
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping, Union


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


# Logging utility functions taken from flair: https://github.com/flairNLP/flair/blob/9b353e74f3ce1450e0441a41004008629d02ffcb/flair/training_utils.py


def init_output_file(base_path: Union[str, Path], file_name: str) -> Path:
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, "w", encoding="utf-8").close()
    return file


def log_line(log):
    log.info("-" * 100)


def add_file_handler(log, output_file):
    init_output_file(output_file.parents[0], output_file.name)
    fh = logging.FileHandler(output_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh
