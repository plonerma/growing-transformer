import logging

import torch
from transformers.models.bert.modeling_bert import BertAttention

from growing_transformer.attention import GrowingAttention

from .base import GrowingTest

log = logging.getLogger("growing_transformer.tests")


class TestMultiheadAttention(GrowingTest):
    model_class = GrowingAttention

    def test_function(self):
        # initialize growing multihead attention block
        config = self.new_config()
        growing_model = self.new_model(config)

        # get state from growing model
        state = growing_model.state_dict()

        log.info("State:")
        for k, v in state.items():
            log.info(f"{k}: {v.size()}")

        bert_attention = BertAttention(config)

        assert bert_attention.state_dict().keys() == state.keys()

        # load state for growing model into torch model
        bert_attention.load_state_dict(state)

        # compare the function of the two transformers
        bert_attention.eval()
        growing_model.eval()

        x = self.random_batch()
        # x = torch.rand(batches, length, embed_dim)

        y_a, attn_a = growing_model(x, return_attention=True)
        y_b, attn_b = bert_attention(x, output_attentions=True)

        assert torch.all(torch.abs(attn_a - attn_b) < 1e-5)
        assert torch.all(torch.abs(y_a - y_b) < 1e-5)
