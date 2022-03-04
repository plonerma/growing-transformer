""" When growing with an extremly small step size, the model should maintain its
function. """

import pytest
import torch
from .util import with_test_models, eps


@pytest.mark.parametrize("grow_params", [
    dict(split=True, num_novel=4, eps_split=1, eps_novel=1, step_size=1e-7),
    dict(split=False, num_novel=4, eps_split=1, eps_novel=1e-7, step_size=1),
    dict(split=True, num_novel=0, eps_split=1e-7, eps_novel=1, step_size=1),
    dict(split=True, num_novel=4, eps_split=1e-7, eps_novel=1e-7, step_size=1),
])
@with_test_models
def test_growth(grow_params, model_spec):
    """ With these growth parameters, the function of the network should be
        changed (only) a litte bit.
    """

    torch.manual_seed(0)

    model_type, model_args = model_spec

    config = dict(grow_params)
    step_size = config.pop('step_size')
    model = model_type(*model_args, config=config)

    x = torch.rand(64, 10, model.in_features)

    y_a = model(x)

    model.grow(step_size)

    y_b = model(x)

    diff = torch.abs(y_a - y_b)

    # the change should be minor
    assert torch.all(diff < eps)

    # but should be there
    assert not torch.all(diff > eps*eps)
