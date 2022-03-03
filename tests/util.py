import pytest

from growing import MLP, ScaledDotProductAttention, MultiheadAttention


eps = 1e-5

test_models = [(MLP, (16, 8, 12)), (ScaledDotProductAttention, (16, 5, 7)), (MultiheadAttention, (16, 5, 7))]


def model_spec_id(spec):
    cls, args = spec
    params = ', '.join([str(s) for s in args])
    return f"{cls.__name__}({params})"


def with_test_models(f):
    return pytest.mark.parametrize("model_spec", test_models, ids=model_spec_id)(f)
