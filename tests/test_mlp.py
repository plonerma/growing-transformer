from growing_transformer.mlp import GrowingMLP

from .base import SplittingTest


class TestMLP(SplittingTest):
    def new_model(self, config):
        return GrowingMLP(self.embed_dim, 8, 12, config=config)
