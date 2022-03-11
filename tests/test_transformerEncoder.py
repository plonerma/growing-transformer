import logging

import torch
from transformers.models.bert.modeling_bert import BertEncoder

from growing_transformer.encoder import GrowingEncoder

from .base import GrowingTest

log = logging.getLogger("growing_transformer.tests")


class TestTransformerEncoder(GrowingTest):
    model_class = GrowingEncoder

    def new_model(self, config={}):
        model = super().new_model(config)

        # initialize layer norms with differen values
        for name, param in model.named_parameters():
            if name.endswith("layer_norm.weight"):
                torch.nn.init.uniform_(param, -1, 1)
            elif name.endswith("layer_norm.bias"):
                torch.nn.init.uniform_(param, -0.2, 0.2)

        return model

    def test_function(self):
        # initialize growing multihead attention block
        config = self.new_config()
        growing_model = self.new_model(config)

        # get state from growing model
        state = growing_model.state_dict()

        log.info("Growing encoder state:")
        for k, v in state.items():
            log.info(f"- {k}: {v.size()}")

        bert_encoder = BertEncoder(config)

        log.info("Bert encoder state:")
        for k, v in bert_encoder.state_dict().items():
            log.info(f"- {k}: {v.size()}")

        assert bert_encoder.state_dict().keys() == state.keys()

        # load state for growing model into torch model
        bert_encoder.load_state_dict(state)

        # compare the function of the two transformers
        bert_encoder.eval()
        growing_model.eval()

        x = self.random_batch()

        y_a = growing_model(x)
        y_b = bert_encoder(x)

        y_b = y_b.last_hidden_state

        diff = torch.abs(y_a - y_b)

        log.info(f"Max. difference: {diff.max()}")

        assert torch.all(diff < 1e-5)
