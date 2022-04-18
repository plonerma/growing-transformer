""" When growing with an extremly small step size, the model should maintain its
function. """

import torch

from growing_transformer import GrowingConfig
from growing_transformer.model import AttentionOutput

from .base import GrowingTest


class TestAttentionOutput(GrowingTest):
    model_class = AttentionOutput

    def new_model(self, config={}):
        if not isinstance(config, GrowingConfig):
            config = self.new_config(config)
        model = self.model_class(config)

        heads = config.num_attention_heads

        # monkey-patch attention method to provide attention matrix
        original_forward = model.forward

        def patched_forward(x):
            # "calculate" random attention
            torch.manual_seed(0)
            batch_size, length, _ = x.size()

            attention = torch.rand(batch_size, heads, length, length)
            attention = torch.softmax(attention, dim=-1)

            # call original funcion
            return original_forward(x, attention)

        model.forward = patched_forward
        return model
