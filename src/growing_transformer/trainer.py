import logging
from contextlib import contextmanager
from typing import Optional

import torch

# https://github.com/pytorch/pytorch/issues/39009
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import DataLoader

from .train_util import log_line

log = logging.getLogger("growing_transformer")


class Trainer:
    def __init__(self, model, criterion, grow_func=None):
        self.model = model
        self.criterion = criterion
        self.grow_func = grow_func

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

            for train_x, train_y in batch_loader:
                optimizer.zero_grad()

                y = self.model(train_x)
                loss = self.criterion(y, train_y)

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

            for train_x, train_y in batch_loader:
                y = self.model(train_x)
                loss = self.criterion(y, train_y)
                loss.backward()

    def train(
        self,
        train_data,
        *,
        learning_rate: float = 0.01,
        use_onecycle: bool = True,
        num_epochs: int = 5,
        growth_phases: int = 5,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        propagate_interrupt=False,
        tensorboard_writer=None,
        **kw,
    ):

        log.info(f"Model: {self.model}")
        log_line(log)
        log.info("Parameters:")
        log.info(f" - learning_rate: {learning_rate}")
        log.info(f" - use_onecycle: {use_onecycle}")
        log.info(f" - epochs per growth phase: {num_epochs}")
        log.info(f" - growth phases: {growth_phases}")
        log_line(log)

        use_tensorboard = tensorboard_writer is not None

        log_line(log)

        try:
            total_epochs = 0

            for growth_phase in range(growth_phases):
                if growth_phase > 0:
                    if self.grow_func is not None:
                        self.grow_func(self, self.model, growth_phase)

                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

                if use_onecycle:
                    num_batches = len(train_data) // batch_size + 1

                    scheduler = OneCycleLR(
                        optimizer, steps_per_epoch=num_batches, epochs=num_epochs, max_lr=learning_rate
                    )
                else:
                    scheduler = None

                for epoch in range(num_epochs):
                    loss_sum = 0

                    batch_loader = DataLoader(
                        train_data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0 if num_workers is None else num_workers,
                    )

                    for train_x, train_y in batch_loader:

                        optimizer.zero_grad()

                        y = self.model(train_x)
                        loss = self.criterion(y, train_y)

                        loss.backward()
                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        loss_sum += loss.data

                    loss = loss_sum / len(batch_loader)

                    if use_tensorboard:
                        lr = 0.0
                        for group in optimizer.param_groups:
                            lr = group["lr"]

                        tensorboard_writer.add_scalar("training/learning rate", lr, total_epochs)
                        tensorboard_writer.add_scalar("training/train loss", loss, total_epochs)
                        tensorboard_writer.add_scalar(
                            "training/model size", sum(p.numel() for p in self.model.parameters()), total_epochs
                        )

                    total_epochs += 1
                log.info(f"growth phase {growth_phase+1} - {total_epochs} epochs - loss: {loss}")

            if use_tensorboard:
                log.info(f" - learning_rate: {learning_rate}")
                log.info(f" - use_onecycle: {use_onecycle}")
                log.info(f" - epochs per growth phase: {num_epochs}")
                log.info(f" - growth phases: {growth_phases}")

        except KeyboardInterrupt:
            log_line(log)
            log.warning("Exiting from training early.")

            if propagate_interrupt:
                raise KeyboardInterrupt

        log_line(log)
        log.info("Done.")

        return dict(final_train_loss=loss)
