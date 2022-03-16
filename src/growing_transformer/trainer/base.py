import logging
from typing import Optional

import torch

# https://github.com/pytorch/pytorch/issues/39009
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import DataLoader

from .util import log_line

log = logging.getLogger("growing_transformer")


class BaseTrainer:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        train_data,
        learning_rate: float = 0.01,
        use_onecycle: bool = True,
        num_epochs: int = 5,
        growth_phases: int = 5,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        propagate_interrupt=False,
        tensorboard_writer=None,
        log_training_info=True,
        start_epoch=0,
        **kw,
    ):

        if log_training_info:
            log.info(f"Model: {self.model}")
            log_line(log)
            log.info("Parameters:")
            log.info(f" - learning_rate: {learning_rate}")
            log.info(f" - use_onecycle: {use_onecycle}")
            log_line(log)

        use_tensorboard = tensorboard_writer is not None

        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            if use_onecycle:
                num_batches = len(train_data) // batch_size + 1

                scheduler = OneCycleLR(optimizer, steps_per_epoch=num_batches, epochs=num_epochs, max_lr=learning_rate)
            else:
                scheduler = None

            for epoch in range(start_epoch, start_epoch + num_epochs):
                batch_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0 if num_workers is None else num_workers,
                )

                loss_sum = 0

                for batch in batch_loader:
                    optimizer.zero_grad()
                    loss = self.model.forward_loss(batch)
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

                    tensorboard_writer.add_scalar("training/learning rate", lr, epoch)
                    tensorboard_writer.add_scalar("training/train loss", loss, epoch)
                    tensorboard_writer.add_scalar(
                        "training/model size", sum(p.numel() for p in self.model.parameters()), epoch
                    )

        except KeyboardInterrupt:
            log_line(log)
            log.warning("Exiting from training early.")

            if propagate_interrupt:
                raise KeyboardInterrupt

        return dict(final_train_loss=loss, epoch=epoch)
