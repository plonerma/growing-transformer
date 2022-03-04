""" When growing with an extremly small step size, the model should maintain its
function. """

import pytest
import torch
from .util import with_test_models, eps


def select_all(size):
    """Return indices for all possible neurons."""
    if isinstance(size, torch.Size):
        return torch.arange(size.numel())
    return [select_all(s) for s in size]


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=True, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
])
@with_test_models
def test_degrow_to_grown(grow_params, model_spec):
    """ The model should produce the same output before and after degrow
        (after having called grow).
    """
    torch.manual_seed(0)

    model_type, model_args = model_spec

    config = dict(grow_params)
    step_size = config.pop('step_size')
    model = model_type(*model_args, config=config)

    x = torch.rand(64, 10, model.in_features)

    sizes = model.grow(step_size)

    y_a = model(x)

    # degrow keeping all neurons
    model.degrow(select_all(sizes))

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    assert torch.all(diff < eps)


def select_none(size):
    """Return indices for all possible neurons."""
    if isinstance(size, torch.Size):
        return torch.arange(0)
    return [select_none(s) for s in size]


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=4, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=True, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
    dict(split=False, num_novel=0, eps_split=.1, eps_novel=.2, step_size=.3),
])
@with_test_models
def test_degrow_to_original(grow_params, model_spec):
    """Selecting none of the neurons should yield the original model."""
    torch.manual_seed(0)

    model_type, model_args = model_spec

    config = dict(grow_params)
    step_size = config.pop('step_size')
    model = model_type(*model_args, config=config)

    x = torch.rand(64, 10, model.in_features)

    y_a = model(x)

    sizes = model.grow(step_size)
    # degrow deleting all recently grown neurons

    model.degrow(select_none(sizes))

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    assert torch.all(diff < eps)
