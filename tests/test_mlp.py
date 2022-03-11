from growing_transformer.model import GrowingMLP

from .base import SplittingTest


class TestMLP(SplittingTest):
    model_class = GrowingMLP
