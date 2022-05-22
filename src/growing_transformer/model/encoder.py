from typing import Optional

import torch
from torch import Tensor
from torch.nn import ModuleList, Parameter

import growing_transformer

from ..configuration import GrowingConfig
from .base import GrowingModule, NamedDirectionParams
from .layer import GrowingLayer


class GrowingEncoder(GrowingModule):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)
        self.layer = torch.nn.ModuleList([GrowingLayer(config=config) for _ in range(self.config.num_hidden_layers)])

        self.reset_grow_state()
        self.to(growing_transformer.device)

    def reset_grow_state(self) -> None:
        # in this case, it is layers not neurons
        self.step_size: Optional[torch.nn.Parameter] = None

        self._new_layers: ModuleList = ModuleList()

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None):
        for i, layer in enumerate(self.layer):
            x = layer(x, attention_mask=attention_mask)

            if self.step_size is not None:
                x = self._new_layers[i](x, influence_factor=self.step_size[i], attention_mask=attention_mask)
        return x

    def _direction_params(self) -> NamedDirectionParams:
        return {f"direction_{n}": p for n, p in self._new_layers.named_parameters()}

    def grow(self, num_novel: int = 0, split: bool = False) -> torch.Size:
        # one layer in every possible location: between all existing layers + at the start and end
        new_layers = len(self.layer)

        self.step_size = Parameter(
            torch.ones(new_layers, device=growing_transformer.device) * self.config.layer_step_size
        )

        self._new_layers = torch.nn.ModuleList()

        for prev_layer in self.layer:
            new_layer = GrowingLayer(config=self.config)

            if split:
                new_layer.load_state_dict(prev_layer.state_dict())
            else:
                new_layer.apply(new_layer._init_weights)

            new_layer.layer_norm.weight.data = prev_layer.layer_norm.weight
            new_layer.layer_norm.bias.data = prev_layer.layer_norm.bias
            prev_layer.layer_norm._weight = prev_layer.layer_norm.weight.data
            prev_layer.layer_norm._bias = prev_layer.layer_norm.bias.data

            prev_layer.layer_norm.weight.data = torch.ones(prev_layer.layer_norm.weight.size())
            prev_layer.layer_norm.bias.data = torch.zeros(prev_layer.layer_norm.bias.size())

            self._new_layers.append(new_layer)

        self.train(self.training)
        return self.step_size.size()

    def degrow(self, selected):
        # sort in descending order, so we can add layers without fiddeling with indices

        was_selected = torch.zeros(self.step_size.size(), dtype=bool)
        was_selected[selected] = True

        for i in torch.arange(was_selected.size(0) - 1, -1, -1):
            i = int(i)
            if was_selected[i]:
                layer = self._new_layers[i]

                # permanently apply influence factor to layer
                layer.apply_influence_factor(self.step_size[i])

                # tmp variables can be cleared (now part of new layer)
                self.layer[i].layer_norm._weight = None
                self.layer[i].layer_norm._bias = None

                # add new layer to existing module list
                self.layer.insert(i + 1, layer)
            else:
                # restore weight and bias
                layer = self.layer[i]
                layer.layer_norm.weight.data = layer.layer_norm._weight
                layer.layer_norm.bias.data = layer.layer_norm._bias

                layer.layer_norm._weight = None
                layer.layer_norm._bias = None

        # reset temporary variable
        self.reset_grow_state()

        # set training flag of all new modules to own state
        self.train(self.training)

    def update_config(self, num_added: int):
        # set config
        self.config.num_hidden_layers = len(self.layer) + num_added
