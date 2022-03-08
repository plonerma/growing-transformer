""" When growing with an extremly small step size, the model should maintain its
function. """

import torch

from growing_transformer.scaledDotProductAttention import (
    GrowingScaledDotProductAttention,
)

from .base import GrowingTest


class TestScaledDotProductAttention(GrowingTest):
    num_heads = 4
    d_head = 16

    def new_model(self, config):
        return GrowingScaledDotProductAttention(
            self.embed_dim, self.num_heads, self.d_head, batch_first=False, config=config
        )

    def test_function(self):
        model = self.new_model({})
        x = self.random_batch()

        in_proj_bias = torch.cat(
            [
                model.query_linear.bias,
                model.key_linear.bias,
                torch.zeros(self.embed_dim),
            ]
        )

        in_proj_weight = torch.cat(
            [
                model.query_linear.weight,
                model.key_linear.weight,
                torch.eye(self.embed_dim),
            ]
        )

        out_proj_weight = torch.eye(self.embed_dim)
        out_proj_bias = torch.zeros(self.embed_dim)

        attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
            x,
            x,
            x,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            in_proj_bias,
            None,
            None,
            False,
            0.0,
            out_proj_weight,
            out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

        model_output_weights = model(x)

        # multi_head_attention_forward averages attention weights over all heads
        model_output_weights = model_output_weights.mean(axis=1)

        diff = torch.abs(attn_output_weights - model_output_weights)

        assert torch.all(diff < 1e-5)
