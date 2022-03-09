from growing_transformer.mlp import GrowingMLP

from .base import SplittingTest


class TestMLP(SplittingTest):
    model_class = GrowingMLP
