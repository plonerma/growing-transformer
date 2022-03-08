import logging

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from growing_transformer import TransformerLayer

from .base import GrowingTest

log = logging.getLogger("growing_transformer.tests")


class TestTransformerLayer(GrowingTest):
    embed_dim = 64
    hidden_size = 128
    batches = 16
    length = 32

    num_heads = 4
    d_head = 16

    def new_model(self, config):
        config = {**config, "bert_state_dict": True, "intermediate_act_fn": "gelu"}
        return TransformerLayer(self.embed_dim, self.num_heads, self.d_head, self.hidden_size, config=config)

    def test_function(self):
        # initialize growing multihead attention block
        growing_model = self.new_model({})

        # bert multihead attention block
        configuration = BertConfig(
            hidden_size=self.embed_dim,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_size,
            hidden_act="gelu"
        )

        # get state from growing model
        state = growing_model.state_dict()

        log.info(" Growing layer state:")
        for k, v in state.items():
            log.info(f"- {k}: {v.size()}")

        bert_layer = BertLayer(configuration)

        log.info("Bert layer state:")
        for k, v in bert_layer.state_dict().items():
            log.info(f"- {k}: {v.size()}")

        assert bert_layer.state_dict().keys() == state.keys()

        # load state for growing model into torch model
        bert_layer.load_state_dict(state)

        # compare the function of the two transformers
        bert_layer.eval()
        growing_model.eval()

        x = self.random_batch()

        y_a = growing_model(x)
        y_b, = bert_layer(x)

        print(y_b)

        assert torch.all(torch.abs(y_a - y_b) < 1e-5)
