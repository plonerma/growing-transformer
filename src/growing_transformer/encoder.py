import torch

from .base import Growing
from .configuration import GrowingConfig
from .layer import GrowingLayer


class GrowingEncoder(Growing):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)
        self.layer = torch.nn.ModuleList([GrowingLayer(config=config) for _ in range(self.config.num_hidden_layers)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    # TODO: Implement depth growth
