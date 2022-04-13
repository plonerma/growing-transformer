import logging
import re
import time
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

    def grow_model(self, substeps, grow_data=None, tensorboard_writer=None, index=None):
        grown_modules = list()

        for params in substeps:

            if params.get("match_end", True):
                pattern = re.compile(params["match"] + "$")
            else:
                pattern = re.compile(params["match"])

            num_novel = params.get("num_novel", 0)
            split = params.get("split", False)
            num_kept_parts = params.get("num_kept_parts", 0)

            for name, m in self.model.growing_modules():
                if re.search(pattern, name) is not None:
                    size = m.grow(num_novel=num_novel, split=split)

                    grown_modules.append((m, size, num_kept_parts))

        if self._tune_direction:
            assert grow_data is not None
            self.tune_direction(grow_data)

        if self._tune_direction:
            assert grow_data is not None
            self.tune_new_parts(grow_data)

        if self._selection_method == "firefly":
            self.calculate_new_gradient(grow_data)

            for m, size, num_kept_parts in grown_modules:
                selected = m.select(num_kept_parts)
                m.degrow(selected)
                if tensorboard_writer is not None and selected.numel() and index is not None:
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, index)

        elif self._selection_method == "random":
            for m, size, num_kept_parts in grown_modules:

                if len(size) == 0:
                    continue
                *shape, n_neurons = size

                selected = torch.stack(
                    [torch.randperm(n_neurons)[:num_kept_parts] for _ in range(size.numel() // n_neurons)]
                )

                selected = selected.reshape(*shape, -1)

                m.degrow(selected)
                if tensorboard_writer is not None and selected.numel() and index is not None:
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, index)

    def train(  # type: ignore[override]
        self,
        train_data,
        schedule: GrowthSchedule,
        grow_data=None,
        test_data=None,
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
            log.info(f" - growth phases: {len(schedule)}")
            log_line(log)

        global_step = 0
        current_epoch = 0

        for step_index, (step_type, step_params) in enumerate(schedule):
            if step_type == step_type.grow:
                grow_start = time.time()
                self.grow_model(
                    step_params, grow_data=grow_data, tensorboard_writer=tensorboard_writer, index=step_index
                )
                grow_end = time.time()

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("time/growth", grow_end - grow_start, global_step)

            elif step_type == step_type.train:
                train_params = {**kw, **step_params}

                train_start = time.time()

                train_info = super().train(
                    train_data=train_data,
                    test_data=test_data,
                    num_epochs=step_params,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    propagate_interrupt=propagate_interrupt,
                    tensorboard_writer=tensorboard_writer,
                    log_training_info=False,
                    start_epoch=current_epoch,
                    global_step=global_step,
                    **train_params,
                )
                train_end = time.time()

                current_epoch = train_info["epoch"]
                global_step = train_info["global_step"]

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("time/training", train_end - train_start, global_step)

                log.info(f"Epoch {current_epoch} - loss: {train_info['final_train_loss']}")

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
