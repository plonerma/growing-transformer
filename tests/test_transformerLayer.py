import logging

import torch
from transformers.models.bert.modeling_bert import BertLayer

from growing_transformer.layer import GrowingLayer

from .base import GrowingTest

log = logging.getLogger("growing_transformer.tests")


class TestTransformerLayer(GrowingTest):
    model_class = GrowingLayer

    def test_function(self):
        # initialize growing multihead attention block
        config = self.new_config()
        growing_model = self.model_class(config)

        # get state from growing model
        state = growing_model.state_dict()

        log.info(" Growing layer state:")
        for k, v in state.items():
            log.info(f"- {k}: {v.size()}")

        bert_layer = BertLayer(config)

        log.info("Bert layer state:")
        for k, v in bert_layer.state_dict().items():
            log.info(f"- {k}: {v.size()}")

        assert bert_layer.state_dict().keys() == state.keys()

        # load state for growing model into torch model
        bert_layer.load_state_dict(state)

        # compare the function of the two transformers
        bert_layer.eval()
        growing_model.eval()

        x = self.random_batch(config)

        y_a = growing_model(x)
        (y_b,) = bert_layer(x)

        print(y_b)

        assert torch.all(torch.abs(y_a - y_b) < 1e-5)
