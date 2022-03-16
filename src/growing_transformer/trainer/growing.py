import logging
from contextlib import contextmanager
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .base import BaseTrainer
from .schedule import GrowthSchedule
from .util import log_line

log = logging.getLogger("growing_transformer")


class GrowingTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        *,
        tune_direction: bool = True,
        tune_new_parts: bool = True,
        selection_method="firefly",
    ):
        super().__init__(model)

        self._tune_direction = tune_direction
        self._tune_new_parts = tune_new_parts
        self._selection_method = selection_method

    def grow_model(self, phase, grow_data=None, tensorboard_writer=None):

        # TODO: make sure modules are grown in proper order
        sizes = list()

        for m in self.model.growing_modules():
            sizes.append(m.grow(phase.grow_params(m)))

        if self._tune_direction:
            assert grow_data is not None
            self.tune_direction(grow_data)

        if self._tune_direction:
            assert grow_data is not None
            self.tune_new_parts(grow_data)

        if self._selection_method == "firefly":
            self.calculate_new_gradient(grow_data)

            for m in self.model.growing_modules():
                # TODO: This needs to be adapted to account for different modules
                selected = m.select(phase.num_kept_parts(m))
                m.degrow(selected)
                if tensorboard_writer is not None and selected.numel():
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, phase.index)

        elif self._selection_method == "random":
            for s, m in zip(sizes, self.model.growing_modules()):

                if len(s) == 0:
                    continue
                *shape, n_neurons = s

                selected = torch.stack(
                    [torch.randperm(n_neurons)[: phase.num_kept_neurons(m)] for _ in range(s.numel() // n_neurons)]
                )

                selected = selected.reshape(*shape, -1)

                m.degrow(selected)
                if tensorboard_writer is not None and selected.numel():
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, phase.index)

        # apply config update
        self.model.config.update(phase.config_update)

    def train(  # type: ignore[override]
        self,
        train_data,
        schedule: GrowthSchedule,
        learning_rate: float = 0.01,
        use_onecycle: bool = True,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        propagate_interrupt=False,
        tensorboard_writer=None,
        log_training_info=True,
        **kw,
    ):
        if log_training_info:

            log.info(f"Model: {self.model}")
            log_line(log)
            log.info("Parameters:")
            log.info(f" - learning_rate: {learning_rate}")
            log.info(f" - use_onecycle: {use_onecycle}")
            log.info(f" - growth phases: {len(schedule)}")
            log_line(log)

        epoch = 0

        for phase in schedule:
            if not phase.is_initial:
                self.grow_model(self, phase, tensorboard_writer=tensorboard_writer)

            train_info = super().train(
                train_data=train_data,
                learning_rate=learning_rate,
                use_onecycle=use_onecycle,
                num_epochs=phase.epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                propagate_interrupt=propagate_interrupt,
                tensorboard_writer=tensorboard_writer,
                log_training_info=False,
                start_epoch=epoch,
                **kw,
            )

            epoch = train_info["epoch"]

            log.info(f"growth phase {phase.index} - {epoch} epochs - loss: {train_info['final_train_loss']}")

        return train_info

    @contextmanager
    def some_grad_only(self, *some_parameters):
        # temporarily save requires_grad for all parameters
        _requires_grad = [p.requires_grad for p in self.model.parameters()]

        # disable all grads
        for p in self.model.parameters():
            p.requires_grad = False

        # enable grads some parameters
        for p in some_parameters:
            if p is not None:
                p.requires_grad = True

        yield  # yield for forward pass

        # reset requires_grad of all parameters
        for p, rg in zip(self.model.parameters(), _requires_grad):
            p.requires_grad = rg

    def tune_with_penalty(
        self,
        params,
        train_data,
        optimizer=torch.optim.RMSprop,
        optim_kw={},
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            optim_kw = {"lr": 1e-3, "momentum": 0.1, "alpha": 0.9, **optim_kw}

            optimizer = optimizer(params, **optim_kw)

        with self.some_grad_only(*params):
            batch_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0 if num_workers is None else num_workers,
            )

            for batch in batch_loader:
                optimizer.zero_grad()

                loss = self.model.forward_loss(batch)

                penalty = torch.tensor(0.0)
                for p in params:
                    penalty += (p**2).sum()

                loss.backward()
                penalty.backward()
                optimizer.step()

    def tune_direction(self, *args, **kwargs):
        self.tune_with_penalty(list(self.model.direction_params()), *args, **kwargs)

    def tune_new_parts(self, *args, **kwargs):
        self.tune_with_penalty(list(self.model.new_params()), *args, **kwargs)

    def calculate_new_gradient(
        self, train_data, batch_size: int = 32, shuffle: bool = True, num_workers: Optional[int] = None
    ):

        batch_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0 if num_workers is None else num_workers,
        )

        with self.some_grad_only(*self.model.new_params()):
            self.model.zero_grad()

            for batch in batch_loader:
                loss = self.model.forward_loss(batch)
                loss.backward()
