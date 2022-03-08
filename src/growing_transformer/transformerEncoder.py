from typing import Any, Mapping

import torch

from .base import Growing
from .transformerLayer import TransformerLayer


class TransformerEncoder(Growing):
    def __init__(self, embed_dim, num_heads, d_head, hidden_size, *, config: Mapping[str, Any]):
        super().__init__(config=config)
        self.layer = torch.nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, d_head, hidden_size, config=config)
                for _ in range(self.get_config("num_hidden_layers", default=6))
            ]
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    # TODO: Implement depth growth
