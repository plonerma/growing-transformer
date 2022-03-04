""" When growing with an extremly small step size, the model should maintain its
function. """

import pytest
import torch
from .util import eps

from growing import ScaledDotProductAttention


def test_scaled_dot_product():
    embed_dim = 64
    num_heads = 4
    d_head = 16

    batches = 64
    length = 512

    x = torch.rand(length, batches, embed_dim)

    dp = ScaledDotProductAttention(embed_dim, num_heads, d_head, batch_first=False)


    in_proj_bias = torch.cat([
        dp.query_linear.bias,
        dp.key_linear.bias,
        torch.zeros(embed_dim),
    ])

    in_proj_weight = torch.cat([
        dp.query_linear.weight,
        dp.key_linear.weight,
        torch.eye(embed_dim),
    ])

    out_proj_weight = torch.eye(embed_dim)
    out_proj_bias = torch.zeros(embed_dim)


    attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
                x, x, x, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                None, None, False,
                0.0, out_proj_weight, out_proj_bias,
                training=False,
                key_padding_mask=None, need_weights=True,
                attn_mask=None)


    dp_output_weights = dp(x)

    # multi_head_attention_forward averages attention weights over all heads
    dp_output_weights = dp_output_weights.mean(axis=1)

    diff = torch.abs(attn_output_weights - dp_output_weights)

    assert torch.all(diff < eps)
