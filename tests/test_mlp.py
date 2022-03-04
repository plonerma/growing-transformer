import pytest

from growing import MLP

from .base import SplittingTest


class TestMLP(SplittingTest):
    def new_model(self, config):
        return MLP(self.embed_dim, 8, 12, config=config)
