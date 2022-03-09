import logging
from typing import List, Optional, Type

import pytest
import torch

from growing_transformer import Growing, GrowingConfig, GrowingModule

log = logging.getLogger("growing_transformer.tests")


class GrowingTest:
    num_batches = 16
    length = 32

    base_config = dict(
        hidden_act="gelu",
        d_model=64,
        num_heads=4,
        d_head=16,
        intermediate_size=128,
        bert_like_state_dict=True,
    )

    model_class: Optional[Type[Growing]] = None

    growth_params = [
        dict(split=False, num_novel=4, eps_novel=1, step_size=1e-5),
        dict(split=False, num_novel=4, eps_novel=1e-5, step_size=1),
    ]

    degrowth_params = [
        dict(split=True, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=False, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=True, num_novel=0, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=False, num_novel=0, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=True, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=1.0),
    ]

    def random_batch(self, config):
        return torch.rand(self.num_batches, self.length, config.d_model)

    def new_config(self, params={}):
        return GrowingConfig(**params, **self.base_config)

    def test_state_loading(self):
        config = self.new_config({})
        model_a = self.model_class(config)
        model_b = self.model_class(config)

        state = model_a.state_dict()
        model_b.load_state_dict(state)

        model_a.eval()
        model_b.eval()

        x = self.random_batch(config)
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

        torch.manual_seed(0)

        config = self.new_config(params)
        model = self.model_class(config)

        model.eval()

        x = self.random_batch(config)

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow()
        else:
            for m in model.growing_modules():
                m.grow()

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

        config = self.new_config(params)
        model = self.model_class(config)

        model.eval()

        x = self.random_batch(config)

        y_a = model(x)

        if isinstance(model, GrowingModule):
            # grow only the module itself
            model.grow()
        else:
            # test for composite growth
            for m in model.growing_modules():
                m.grow()

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

        config = self.new_config(params)
        model = self.model_class(config)

        model.eval()

        x = self.random_batch(config)

        modules: List[Growing]
        if isinstance(model, GrowingModule):
            modules = [model]
        else:
            modules = list(model.growing_modules())

        sizes: List[torch.Size] = [m.grow() for m in modules]

        y_a = model(x)

        # degrow keeping all neurons
        for m, size in zip(modules, sizes):
            m.degrow(torch.arange(size.numel()))

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

        config = self.new_config(params)
        model = self.model_class(config)

        model.eval()

        x = self.random_batch(config)

        y_a = model(x)

        modules: List[Growing]
        if isinstance(model, GrowingModule):
            modules = [model]
        else:
            modules = list(model.growing_modules())

        sizes: List[torch.Size] = [m.grow() for m in modules]

        # degrow deleting all recently grown neurons
        for m, size in zip(modules, sizes):
            m.degrow(torch.arange(0))

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        assert torch.all(diff < 1e-5)


class SplittingTest(GrowingTest):
    growth_params = [
        dict(split=True, num_novel=4, eps_split=1, eps_novel=1, step_size=1e-5),
        dict(split=False, num_novel=4, eps_split=1, eps_novel=1e-7, step_size=1),
        dict(split=True, num_novel=0, eps_split=1e-4, eps_novel=1, step_size=1),
        dict(split=True, num_novel=4, eps_split=1e-4, eps_novel=1e-4, step_size=1),
    ]

    @pytest.mark.parametrize("params", growth_params)
    def test_growth_is_minor(self, params):
        """With these growth parameters, the function of the network should be
        changed only a litte bit.
        """

        torch.manual_seed(0)

        config = self.new_config(params)
        model = self.model_class(config)

        model.eval()

        x = self.random_batch(config)

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow()
        else:
            for m in model.growing_modules():
                m.grow()

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        log.info(f"Max difference {diff.max()}")

        # the change should be minor
        assert torch.all(diff < 1e-2)
