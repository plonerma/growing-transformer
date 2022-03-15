from typing import Optional

import torch
from torch import BoolTensor, Tensor
from torch.nn import ModuleList, Parameter
from torch.nn.init import uniform_

from ..configuration import GrowingConfig
from .base import GrowingModule
from .layer import GrowingLayer


class GrowingEncoder(GrowingModule):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)
        self.layer = torch.nn.ModuleList([GrowingLayer(config=config) for _ in range(self.config.num_hidden_layers)])

        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # in this case, it is layers not neurons
        self.new_parts: Optional[torch.nn.Parameter] = None

        self._new_layers: ModuleList = ModuleList()

    def forward(self, x: Tensor, attention_mask: Optional[BoolTensor] = None):
        for i, layer in enumerate(self.layer):
            x = layer(x, attention_mask=attention_mask)

            if self.new_parts is not None:
                x = self._new_layers[i](x, influence_factor=self.new_parts[i], attention_mask=attention_mask)
        return x

    def _direction_params(self):
        return list(self._new_layers.parameters())

    def grow(self):
        # one layer in every possible location: between all existing layers + at the start and end
        new_layers = len(self.layer)

        step_size = self.config.layer_step_size
        self.new_parts = Parameter(torch.ones(new_layers) * step_size, requires_grad=False)

        self._new_layers = torch.nn.ModuleList()

        for prev_layer in self.layer:
            new_layer = GrowingLayer(config=self.config)

            eps_weight = self.config.eps_novel_weight
            eps_bias = self.config.eps_novel_bias

            uniform_(new_layer.mlp.linear_out.weight, -eps_weight, eps_weight)
            uniform_(new_layer.mlp.linear_out.bias, -eps_bias, eps_bias)

            uniform_(new_layer.attention.output_linear.weight, -eps_weight, eps_weight)
            uniform_(new_layer.attention.output_linear.bias, -eps_bias, eps_bias)

            new_layer.layer_norm.weight.data = prev_layer.layer_norm.weight
            new_layer.layer_norm.bias.data = prev_layer.layer_norm.bias
            prev_layer.layer_norm._weight = prev_layer.layer_norm.weight.data
            prev_layer.layer_norm._bias = prev_layer.layer_norm.bias.data

            prev_layer.layer_norm.weight.data = torch.ones(prev_layer.layer_norm.weight.size())
            prev_layer.layer_norm.bias.data = torch.zeros(prev_layer.layer_norm.bias.size())

            self._new_layers.append(new_layer)

        self.train(self.training)
        return self.new_parts.size()

    def degrow(self, selected):
        # sort in descending order, so we can add layers without fiddeling with indices

        was_selected = torch.zeros(self.new_parts.size(), dtype=bool)
        was_selected[selected] = True

        for i in torch.arange(was_selected.size(0) - 1, -1, -1):
            i = int(i)
            if was_selected[i]:
                layer = self._new_layers[i]

                # permanently apply influence factor to layer
                layer.apply_influence_factor(self.new_parts[i])

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

        self.train(self.training)
        self.reset_grow_state()
