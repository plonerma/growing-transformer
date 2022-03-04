import pytest
import torch

from .util import eps
from growing import MultiheadAttention

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention


eps = 1e-5


def test_multihead_attention():
    embed_dim = 768
    batches = 16
    length = 512

    # initialize growing multihead attention block
    growing_model = MultiheadAttention(embed_dim, 12, 64)

    # get state from that model
    state = growing_model.bert_state_dict()

    # bert multihead attention block
    configuration = BertConfig()

    print(configuration)

    bert_attention = BertAttention(configuration)

    assert (bert_attention.state_dict().keys() == state.keys())

    # load state for growing model into torch model
    bert_attention.load_state_dict(state)

    # compare the function of the two transformers
    bert_attention.eval()
    growing_model.eval()


    x = torch.zeros(1, 4, embed_dim)
    #x = torch.rand(batches, length, embed_dim)

    y_a, attn_a = growing_model(x, return_attention=True)
    y_b, attn_b = bert_attention(x, output_attentions=True)

    assert torch.all(torch.abs(attn_a - attn_b) < eps)

    print("-----")
    print(attn_a)
    print(y_a)
    print(y_b)
    print(y_a - y_b)

    assert torch.all(torch.abs(y_a - y_b) < eps)
