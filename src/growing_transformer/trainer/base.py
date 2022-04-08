import logging
from typing import Optional, Tuple

import torch

# https://github.com/pytorch/pytorch/issues/39009
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import DataLoader

from .. import device
from .util import log_line

log = logging.getLogger("growing_transformer")


class BaseTrainer:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        train_data,
        test_data=None,
        max_lr: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-06,
        weight_decay: float = 0.01,
        gca_batches: int = 16,  # gradient accumulation batches
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

        self.model.to(device)

        if log_training_info:
            log.info(f"Model: {self.model}")
            log_line(log)
            log.info("Parameters:")
            log.info(f" - max_lr: {max_lr}")
            log.info(f" - use_onecycle: {use_onecycle}")
            log_line(log)

        use_tensorboard = tensorboard_writer is not None

        try:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, **kw
            )

            if use_onecycle:
                num_steps = len(train_data) // (batch_size * gca_batches) + 1

                scheduler = OneCycleLR(
                    optimizer, steps_per_epoch=num_steps, pct_start=0.1, epochs=num_epochs, max_lr=max_lr
                )
            else:
                scheduler = None

            self.model.train()

            for epoch in range(start_epoch, start_epoch + num_epochs):
                log.info(f"Epoch #{epoch:02}")

                batch_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0 if num_workers is None else num_workers,
                )

                loss_gca_sum = 0
                loss_epoch_sum = 0

                optimizer.zero_grad()
                for i, batch in enumerate(batch_loader):

                    loss = self.model.forward_loss(batch)
                    loss.backward()

                    loss_epoch_sum += loss.detach()
                    loss_gca_sum += loss.detach()

                    global_step = (epoch * len(batch_loader) + i) / gca_batches

                    if (i + 1) % gca_batches == 0:

                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        optimizer.zero_grad()

                        tensorboard_writer.add_scalar(
                            "loss/train_loss_accumulated", loss_gca_sum / gca_batches, global_step
                        )

                        loss_gca_sum = 0

                    if i % 1000 == 0 and use_tensorboard:
                        tensorboard_writer.add_scalar("loss/train_batchwise", loss_epoch_sum / (i + 1), global_step)

                loss = loss_epoch_sum / i

                if use_tensorboard:
                    lr = 0.0
                    for group in optimizer.param_groups:
                        lr = group["lr"]

                    tensorboard_writer.add_scalar("learning rate", lr, global_step)
                    tensorboard_writer.add_scalar("loss/train_epoch_avg", loss, global_step)
                    tensorboard_writer.add_scalar(
                        "training/model size", sum(p.numel() for p in self.model.parameters()), global_step
                    )

                log.info(f"Train loss: {loss:.4}")
                if test_data:
                    eval_results = self.evaluate(test_data, batch_size=batch_size)
                    log.info(f"Eval loss: {eval_results['eval_loss']:.4}")
                    log.info(f"Eval accuracy: {eval_results['accuracy']:.4}")
                    log.info(f"Eval perplexity: {eval_results['perplexity']:.4}")
                    if use_tensorboard:
                        tensorboard_writer.add_scalar("loss/evaluation", eval_results["eval_loss"], global_step)
                        tensorboard_writer.add_scalar("accuracy", eval_results["accuracy"], global_step)
                        tensorboard_writer.add_scalar("perplexity", eval_results["perplexity"], global_step)

        except KeyboardInterrupt:
            log_line(log)
            log.warning("Exiting from training early.")

            if propagate_interrupt:
                raise KeyboardInterrupt

        results = dict(final_train_loss=loss, epoch=epoch, total_steps=global_step, **eval_results)

        if test_data:
            results.update(eval_results)

        return results

    def evaluate(self, data, batch_size=32):
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            return self.model.evaluate(data, batch_size=batch_size)
