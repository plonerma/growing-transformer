from itertools import product


def log_line(log):
    log.info("-" * 100)


class GridSearch:
    def __init__(self, grid):
        self.grid = grid

    def __iter__(self):
        keys, values = zip(*self.grid.items())
        return (dict(zip(keys, p)) for p in product(*values))

    def __len__(self):
        l = 1
        for v in self.grid.values():
            l *= len(v)
        return l
