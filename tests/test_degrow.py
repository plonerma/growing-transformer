""" When growing with an extremly small step size, the model should maintain its
function. """

import pytest

import torch

from growing import MLP, ScaledDotProductAttention, MultiheadAttention


eps = 1e-5

test_models = [(MLP, (16, 8, 12)), (ScaledDotProductAttention, (16, 5, 7)), (MultiheadAttention, (16, 5, 7))]


def select_all(size):
    """Return indices for all possible neurons."""
    if isinstance(size, torch.Size):
        return torch.arange(size.numel())
    try:
        return [select_all(s) for s in size]
    except TypeError:
        return torch.arange(size)


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=True, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
])
@pytest.mark.parametrize("model_spec", test_models)
def test_degrow(grow_params, model_spec):
    """ The model should produce the same output before and after degrow
        (after having called grow).
    """
    torch.manual_seed(0)

    model_type, model_args = model_spec

    model = model_type(*model_args)

    x = torch.rand(64, 10, model.in_features)

    sizes = model.grow(**grow_params)

    y_a = model(x)

    # degrow keeping all neurons
    model.degrow(select_all(sizes))

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    assert torch.all(diff < eps)


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=True, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
])
@pytest.mark.parametrize("model_spec", test_models)
def test_degrow_completely(grow_params, model_spec):
    """Selecting none of the neurons should yield the original model."""
    torch.manual_seed(0)

    model_type, model_args = model_spec

    model = model_type(*model_args)

    x = torch.rand(64, 10, model.in_features)

    y_a = model(x)

    model.grow(**grow_params)
    # degrow deleting all recently grown neurons
    model.degrow(torch.arange(0))

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    assert torch.all(diff < eps)
