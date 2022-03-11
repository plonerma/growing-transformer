""" When growing with an extremly small step size, the model should maintain its
function. """

import torch

from growing_transformer.attention import ScaledDotProductAttention

from .base import GrowingTest


class TestScaledDotProductAttention(GrowingTest):
    model_class = ScaledDotProductAttention

    def test_function(self):
        config = self.new_config()
        model = self.new_model(config)

        x = self.random_batch()

        in_proj_bias = torch.cat(
            [
                model.query_linear.bias,
                model.key_linear.bias,
                torch.zeros(config.d_model),
            ]
        )

        in_proj_weight = torch.cat(
            [
                model.query_linear.weight,
                model.key_linear.weight,
                torch.eye(config.d_model),
            ]
        )

        out_proj_weight = torch.eye(config.d_model)
        out_proj_bias = torch.zeros(config.d_model)

        attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
            # function expects (length, batch, ...) tensor
            x.transpose(1, 0),
            x.transpose(1, 0),
            x.transpose(1, 0),
            config.d_model,
            config.num_heads,
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
