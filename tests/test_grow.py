""" When growing with an extremly small step size, the model should maintain its
function. """

import pytest

import torch

from growing import MLP, ScaledDotProductAttention, MultiheadAttention


eps = 1e-5

test_models = [(MLP, (16, 8, 12)), (ScaledDotProductAttention, (16, 5, 7)), (MultiheadAttention, (16, 5, 7))]


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=1, eps_novel=1, step_size=1e-7),
    dict(split=False, num_novel=4, eps_split=1, eps_novel=1e-7, step_size=1),
    dict(split=True, num_novel=0, eps_split=1e-7, eps_novel=1, step_size=1),
    dict(split=True, num_novel=4, eps_split=1e-7, eps_novel=1e-7, step_size=1),
])
@pytest.mark.parametrize("model_spec", test_models)
def test_growth(grow_params, model_spec):
    """ With these growth parameters, the function of the network should be
        changed (only) a litte bit.
    """

    torch.manual_seed(0)

    model_type, model_args = model_spec

    model = model_type(*model_args)

    x = torch.rand(64, 10, model.in_features)

    y_a = model(x)

    model.grow(**grow_params)

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    # the change should be minor
    assert torch.all(diff < eps)

    # but should be there
    assert not torch.all(diff > eps*eps)
