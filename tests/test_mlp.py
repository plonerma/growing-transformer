import logging

import torch
from transformers.models.bert.modeling_bert import BertIntermediate

from growing_transformer.model import GrowingMLP

from .base import SplittingTest

log = logging.getLogger("growing_transformer.tests")


class BertFFN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.linear_out = torch.nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        intermediate_output = self.intermediate(x)
        return self.linear_out(intermediate_output)


class TestMLP(SplittingTest):
    model_class = GrowingMLP

    def test_function(self):
        # initialize growing multihead attention block
        config = self.new_config()
        growing_model = self.new_model(config)

        def replace_key(k):
            ks = k.split(".")

            if ks[0] == "linear_in":
                ks[0] = "intermediate.dense"

            return ".".join(ks)

        # get state from growing model
        state = {replace_key(k): v for k, v in growing_model.state_dict().items()}

        log.info("State of growing model:")
        for k, v in state.items():
            log.info(f"{k}: {v.size()}")

        bert_model = BertFFN(config)
        bert_state = bert_model.state_dict()

        log.info("State of bert model:")
        for k, v in bert_state.items():
            log.info(f"{k}: {v.size()}")

        assert bert_state.keys() == state.keys()

        # load state for growing model into torch model
        bert_model.load_state_dict(state)

        # compare the function of the two transformers
        bert_model.eval()
        growing_model.eval()

        x = self.random_batch()
        # x = torch.rand(batches, length, embed_dim)

        y_a = growing_model(x)
        y_b = bert_model(x)

        assert torch.all(torch.abs(y_a - y_b) < 1e-5)
