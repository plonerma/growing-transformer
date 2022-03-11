import logging

import pytest
import torch
from transformers.models.bert.modeling_bert import BertLayer

from growing_transformer.model import GrowingLayer

from .base import GrowingTest

log = logging.getLogger("growing_transformer.tests")


class TestTransformerLayer(GrowingTest):
    model_class = GrowingLayer

    def test_function(self):
        # initialize growing multihead attention block
        config = self.new_config()
        growing_model = self.new_model(config)

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

        x = self.random_batch()

        y_a = growing_model(x)
        (y_b,) = bert_layer(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max. difference: {diff.max()}")

        assert torch.all(diff < 1e-5)

    @pytest.mark.parametrize("factor", [4.0, 1.0, 1e-1, 1e-2, 1e-5])
    def test_influence_factor(self, factor):
        model = self.new_model()
        model.eval()

        x = self.random_batch()
        y_a = model(x, influence_factor=factor)

        model.apply_influence_factor(factor)
        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max. difference: {diff.max()}")

        assert torch.all(diff < 1e-5)
