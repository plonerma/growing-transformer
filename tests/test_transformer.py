import pytest

from growing import GrowingTransformer
from transformers import BertConfig, BertModel


eps = 1e-5


@pytest.mark.skip(reason="not implemented yet")
def test_transformer_function():
    # initialize growing transformer
    growing_model = GrowingTransformer()

    # get state from that model
    state = growing_model.state_dict()

    # initialize bert transformer
    configuration = BertConfig()
    bert_model = BertModel(configuration)

    # load state for growing model into bert model
    bert_model.load_state_dict(state)

    # compare the function of the two transformers
    bert_model.eval()
    growing_model.eval()

    embed_dim = 768
    batches = 16
    length = 512

    x = torch.rand(length, batches, embed_dim)

    y_a = growing_model(x)
    y_b = bert_model(x)

    assert torch.all(torch.abs(y_a - y_b) < eps)
