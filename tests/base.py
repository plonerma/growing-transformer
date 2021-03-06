import logging
from typing import List, Optional, Type

import pytest
import torch

from growing_transformer import Growing, GrowingConfig, GrowingModule

log = logging.getLogger("growing_transformer.tests")

log.setLevel(logging.DEBUG)


class GrowingTest:
    num_batches = 17
    length = 29
    d_model = 35

    base_config = dict(
        hidden_act="gelu",
        d_model=d_model,
        num_heads=5,
        d_head=7,
        intermediate_size=128,
        bert_like_state_dict=True,
        num_hidden_layers=1,
    )

    model_class: Optional[Type[Growing]] = None

    growth_params = [
        dict(kw=dict(split=False, num_novel=4), config=dict(initializer_range=1, step_size=1e-6)),
        dict(kw=dict(split=False, num_novel=4), config=dict(initializer_range=1e-5, step_size=1)),
    ]

    degrowth_params = [
        dict(kw=dict(split=True, num_novel=4), config=dict(init_split_range=0.1, initializer_range=0.2, step_size=0.3)),
        dict(
            kw=dict(split=False, num_novel=4), config=dict(init_split_range=0.1, initializer_range=0.2, step_size=0.3)
        ),
        dict(kw=dict(split=True, num_novel=0), config=dict(init_split_range=0.1, initializer_range=0.2, step_size=0.3)),
        dict(
            kw=dict(split=False, num_novel=0), config=dict(init_split_range=0.1, initializer_range=0.2, step_size=0.3)
        ),
        dict(kw=dict(split=True, num_novel=4), config=dict(init_split_range=0.1, initializer_range=0.2, step_size=1.0)),
    ]

    def random_batch(self, size=None):
        return torch.rand(self.num_batches, self.length, size or self.d_model)

    def new_config(self, params={}):
        return GrowingConfig(**params, **self.base_config)

    def new_model(self, config={}):
        """Subtests might overwrite this in order to change initialization."""
        if not isinstance(config, GrowingConfig):
            config = self.new_config(config)
        return self.model_class(config)

    def test_state_loading(self):
        model_a = self.new_model()
        model_b = self.new_model()

        state = model_a.state_dict()
        model_b.load_state_dict(state)

        model_a.eval()
        model_b.eval()

        x = self.random_batch()
        y_a = model_a(x)
        y_b = model_b(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        assert torch.all(diff < 1e-10)

    @pytest.mark.parametrize("params", growth_params)
    def test_growth_is_minor(self, params):
        """With these growth parameters, the function of the network should be
        changed only a litte bit.
        """

        config, kw = params["config"], params["kw"]

        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow(**kw)
        else:
            for m in model.growing_modules():
                m.grow(**kw)

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        # the change should be minor
        assert torch.all(diff < 1e-3)

    @pytest.mark.parametrize("params", growth_params)
    def test_growth_exists(self, params):
        """With these growth parameters, the function of the network should be
        changed.
        """
        torch.manual_seed(0)

        config, kw = params["config"], params["kw"]

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            # grow only the module itself
            model.grow(**kw)
        else:
            # test for composite growth
            for m in model.growing_modules():
                m.grow(**kw)

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        # but should be there
        assert torch.any(diff > 1e-12)

    @pytest.mark.parametrize("params", degrowth_params)
    def test_degrow_to_grown(self, params):
        """The model should produce the same output before and after degrow
        (after having called grow).
        """

        torch.manual_seed(0)
        config, kw = params["config"], params["kw"]

        model = self.new_model(config)

        model.eval()

        assert all([not m.training for m in model.modules()])

        x = self.random_batch()

        modules: List[Growing]
        if isinstance(model, GrowingModule):
            modules = [model]
        else:
            modules = list(reversed(list(model.growing_modules())))

        sizes: List[torch.Size] = list()

        for m in modules:
            size = m.grow(**kw)
            sizes.append(size)

            if isinstance(model, GrowingModule) and size.numel() > 0:
                # change step size
                m.step_size.data = torch.normal(0.0, m.step_size.data)

            if len(size) > 0:
                m.update_config(size[-1])

        assert all([not m.training for m in model.modules()])

        y_a = model(x)

        # degrow keeping all neurons
        for m, size in zip(modules, sizes):
            m.degrow(torch.arange(size.numel()))

        assert all([not m.training for m in model.modules()])

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        assert torch.all(diff < 1e-5)

    @pytest.mark.parametrize("params", degrowth_params)
    def test_degrow_to_original(self, params):
        """The model should produce the same output before and after degrow
        (after having called grow).
        """
        torch.manual_seed(0)

        config, kw = params["config"], params["kw"]

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        modules: List[Growing]
        if isinstance(model, GrowingModule):
            modules = [model]
        else:
            modules = list(reversed(list(model.growing_modules())))

        for m in modules:
            m.grow(**kw)
            m.update_config(0)

        # degrow deleting all recently grown neurons
        for m in modules:
            m.degrow(torch.arange(0))

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        assert torch.all(diff < 1e-5)


class SplittingTest(GrowingTest):
    growth_params = [
        dict(kw=dict(split=True, num_novel=4), config=dict(init_split_range=1, initializer_range=1, step_size=1e-5)),
        dict(kw=dict(split=False, num_novel=4), config=dict(init_split_range=1, initializer_range=1e-4, step_size=1)),
        dict(kw=dict(split=True, num_novel=0), config=dict(init_split_range=1e-4, initializer_range=1, step_size=1)),
        dict(kw=dict(split=True, num_novel=4), config=dict(init_split_range=1e-4, initializer_range=1e-4, step_size=1)),
    ]

    @pytest.mark.parametrize("params", growth_params)
    def test_growth_is_minor(self, params):
        """With these growth parameters, the function of the network should be
        changed only a litte bit.
        """

        config, kw = params["config"], params["kw"]

        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow(**kw)
        else:
            for m in model.growing_modules():
                m.grow(**kw)

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        # the change should be minor
        assert torch.all(diff < 1e-2)

    @pytest.mark.parametrize("params", growth_params)
    def test_growth_exists(self, params):
        """With these growth parameters, the function of the network should be
        changed.
        """
        torch.manual_seed(0)

        config, kw = params["config"], params["kw"]

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            # grow only the module itself
            model.grow(**kw)
        else:
            # test for composite growth
            for m in model.growing_modules():
                m.grow(**kw)

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        # but should be there
        assert torch.any(diff > 1e-12)
