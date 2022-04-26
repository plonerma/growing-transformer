import logging
from typing import Optional, Tuple

import torch

# https://github.com/pytorch/pytorch/issues/39009
from torch.optim.lr_scheduler import LambdaLR  # type: ignore
from torch.utils.data import DataLoader

import growing_transformer

from .util import log_line

log = logging.getLogger("growing_transformer")


class BaseTrainer:
    def __init__(self, model, data_collator=None):
        self.model = model
        self.data_collator = data_collator

    def train(
        self,
        train_data,
        test_data=None,
        max_lr: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-06,
        warmup_pct: float = 0.3,
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
        start_epoch: int = 0,
        global_step: int = 0,
        **kw,
    ):

        self.model.to(growing_transformer.device)

        if log_training_info:
            log.info(f"Model: {self.model}")
            log_line(log)
            log.info("Parameters:")
            log.info(f" - max_lr: {max_lr}")
            log.info(f" - use_onecycle: {use_onecycle}")
            log_line(log)

        use_tensorboard = tensorboard_writer is not None

        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, **kw
            )

            #if use_onecycle:
            num_training_steps = len(train_data) // (batch_size * gca_batches) + 1

            #scheduler = OneCycleLR(
            #    optimizer, steps_per_epoch=num_steps, pct_start=warmup_pct, epochs=num_epochs, max_lr=max_lr
            #)

            num_warmup_steps = int(warmup_pct * num_training_steps)

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )

            scheduler = LambdaLR(optimizer, lr_lambda, -1)
            #else:
            #    scheduler = None

            self.model.train()

            # === START OF TRAINING LOOP ===

            for epoch in range(start_epoch, start_epoch + num_epochs):
                batch_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=self.data_collator,
                    num_workers=0 if num_workers is None else num_workers,
                )

                log.info(f"Epoch #{epoch:02}")

                if use_tensorboard:
                    # write intial values

                    for i, group in enumerate(optimizer.param_groups):
                        tensorboard_writer.add_scalar(f"learning_rate/{i}", group["lr"], global_step)
                    tensorboard_writer.add_scalar("epochs_completed", epoch, global_step)

                loss_gca_sum = 0
                loss_epoch_sum = 0

                optimizer.zero_grad()
                for batch_index, batch in enumerate(batch_loader, start=1):

                    loss = self.model.forward_loss(batch)
                    loss.backward()

                    loss_epoch_sum += loss.detach()
                    loss_gca_sum += loss.detach()

                    if batch_index % gca_batches == 0:
                        global_step += 1

                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        optimizer.zero_grad()

                        if use_tensorboard:
                            tensorboard_writer.add_scalar(
                                "loss/train_loss_batch", loss_gca_sum / gca_batches, global_step
                            )
                        loss_gca_sum = 0

                loss = loss_epoch_sum / batch_index

                if use_tensorboard:
                    for i, group in enumerate(optimizer.param_groups):
                        tensorboard_writer.add_scalar(f"learning_rate/{i}", group["lr"], global_step)

                    tensorboard_writer.add_scalar("epochs_completed", epoch + 1, global_step)

                    tensorboard_writer.add_scalar("loss/train_epoch_epoch", loss, global_step)
                    tensorboard_writer.add_scalar(
                        "model size", sum(p.numel() for p in self.model.parameters()), global_step
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

        results = dict(
            final_train_loss=loss,
            epoch=epoch,
            total_steps=(num_epochs * len(batch_loader)) // gca_batches,
            global_step=global_step,
            **eval_results,
        )

        if test_data:
            results.update(eval_results)

        return results

    def evaluate(self, data, batch_size=32):
        self.model.eval()

        with torch.no_grad():
            batch_loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=0 if num_workers is None else num_workers,
            )
            return self.model.evaluate(batch_loader, batch_size=batch_size)
