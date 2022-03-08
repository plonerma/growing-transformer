from abc import abstractmethod
from typing import List, Tuple

import pytest
import torch

from growing_transformer import Growing, GrowingModule


class GrowingTest:
    embed_dim = 64
    batches = 16
    length = 512

    growth_params = [
        dict(num_novel=4, eps_novel=1, step_size=1e-5),
        dict(num_novel=4, eps_novel=1e-5, step_size=1),
    ]

    degrowth_params = [
        dict(split=True, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=False, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=True, num_novel=0, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=False, num_novel=0, eps_split=0.1, eps_novel=0.2, step_size=0.3),
        dict(split=True, num_novel=4, eps_split=0.1, eps_novel=0.2, step_size=1.0),
    ]

    @abstractmethod
    def new_model(config) -> Tuple[GrowingModule, float]:
        pass

    def random_batch(self):
        return torch.rand(self.batches, self.length, self.embed_dim)

    def test_state_loading(self):
        model_a = self.new_model({})
        model_b = self.new_model({})

        state = model_a.state_dict()
        model_b.load_state_dict(state)

        model_a.eval()
        model_b.eval()

        x = self.random_batch()
        y_a = model_a(x)
        y_b = model_b(x)

        diff = torch.abs(y_a - y_b)

        assert torch.all(diff < 1e-10)

    @pytest.mark.parametrize("config", growth_params)
    def test_growth_is_minor(self, config):
        """With these growth parameters, the function of the network should be
        changed only a litte bit.
        """

        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow()
        else:
            for m in model.growing_modules():
                m.grow()

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        # the change should be minor
        assert torch.all(diff < 1e-2)

    @pytest.mark.parametrize("config", growth_params)
    def test_growth_exists(self, config):
        """With these growth parameters, the function of the network should be
        changed.
        """
        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

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

        # but should be there
        assert torch.any(diff > 1e-12)

    @pytest.mark.parametrize("config", degrowth_params)
    def test_degrow_to_grown(self, config):
        """The model should produce the same output before and after degrow
        (after having called grow).
        """

        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

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

        assert torch.all(diff < 1e-5)

    @pytest.mark.parametrize("config", degrowth_params)
    def test_degrow_to_original(self, config):
        """The model should produce the same output before and after degrow
        (after having called grow).
        """
        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

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

        assert torch.all(diff < 1e-5)


class SplittingTest(GrowingTest):
    growth_params = [
        dict(split=True, num_novel=4, eps_split=1, eps_novel=1, step_size=1e-5),
        dict(split=False, num_novel=4, eps_split=1, eps_novel=1e-7, step_size=1),
        dict(split=True, num_novel=0, eps_split=1e-4, eps_novel=1, step_size=1),
        dict(split=True, num_novel=4, eps_split=1e-4, eps_novel=1e-4, step_size=1),
    ]

    @pytest.mark.parametrize("config", growth_params)
    def test_growth_is_minor(self, config):
        """With these growth parameters, the function of the network should be
        changed only a litte bit.
        """

        torch.manual_seed(0)

        model = self.new_model(config)

        model.eval()

        x = self.random_batch()

        y_a = model(x)

        if isinstance(model, GrowingModule):
            model.grow()
        else:
            for m in model.growing_modules():
                m.grow()

        y_b = model(x)

        diff = torch.abs(y_a - y_b)

        # the change should be minor
        assert torch.all(diff < 1e-2)
