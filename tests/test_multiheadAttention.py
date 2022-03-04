import pytest
import torch

from growing import MultiheadAttention

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention

from .base import GrowingTest


class TestMultiheadAttention(GrowingTest):
    embed_dim = 64
    batches = 16
    length = 32

    num_heads = 4
    d_head = 16

    def new_model(self, config):
        return MultiheadAttention(self.embed_dim, self.num_heads, self.d_head, config=config)


    def test_function(self):
        # initialize growing multihead attention block
        growing_model = self.new_model({})

        # get state from that model
        state = growing_model.bert_state_dict()

        for k, v in state.items():
            print(k, v.size())

        # bert multihead attention block

        configuration = BertConfig(hidden_size=self.embed_dim, num_attention_heads=self.num_heads)

        print(configuration)

        bert_attention = BertAttention(configuration)

        assert (bert_attention.state_dict().keys() == state.keys())

        # load state for growing model into torch model
        bert_attention.load_state_dict(state)

        # compare the function of the two transformers
        bert_attention.eval()
        growing_model.eval()


        x = self.random_batch()
        #x = torch.rand(batches, length, embed_dim)

        y_a, attn_a = growing_model(x, return_attention=True)
        y_b, attn_b = bert_attention(x, output_attentions=True)

        assert torch.all(torch.abs(attn_a - attn_b) < 1e-5)
        assert torch.all(torch.abs(y_a - y_b) < 1e-5)
