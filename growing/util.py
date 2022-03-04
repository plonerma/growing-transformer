import torch
from typing import OrderedDict
from collections import OrderedDict as odict


def map_attention_state(state: OrderedDict[str, torch.Tensor],
                        from_bert: bool = False
                        ) -> OrderedDict[str, torch.Tensor]:
    state_map = {
        'self.query.weight': 'dot_product.query_linear.weight',
        'self.query.bias': 'dot_product.query_linear.bias',
        'self.key.weight': 'dot_product.key_linear.weight',
        'self.key.bias': 'dot_product.key_linear.bias',
        'self.value.weight': 'value_linear.weight',
        'self.value.bias': 'value_linear.bias',
        'output.dense.weight': 'output_linear.weight',
        'output.dense.bias': 'output_linear.bias',
        'output.LayerNorm.weight': 'layer_norm.weight',
        'output.LayerNorm.bias': 'layer_norm.bias',
    }

    new_state: OrderedDict[str, torch.Tensor] = odict()

    for k,v in state_map.items():
        if from_bert:
            new_state[v] = state[k]
        else:
            new_state[k] = state[v]

    return new_state
