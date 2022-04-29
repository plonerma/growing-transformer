import logging
import math
import time
from typing import Dict, Mapping, Optional, Tuple

import datasets
import torch

# https://github.com/pytorch/pytorch/issues/39009
from torch.optim.lr_scheduler import LambdaLR  # type: ignore
from torch.utils.data import DataLoader

import growing_transformer

from .util import log_line

log = logging.getLogger("growing_transformer")


class BaseTrainer:
    def __init__(
        self,
        model,
        data_collator=None,
        metric: Optional[datasets.Metric] = None,
        num_workers: int = None,
        batch_size: int = 16,
    ):
        self.model = model
        self.data_collator = data_collator

        if metric is None:
            metric = datasets.load_metric("accuracy")

        self.metric = metric

        if num_workers is None:
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        self.batch_size = batch_size

    def get_batch_loader(self, data, *, batch_size=None, num_workers=None, shuffle=True):
        if num_workers is None:
            num_workers = self.num_workers

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=num_workers,
        )

    def get_lr_scheduler(self, optimizer, *, warmup_pct, total_steps):
        num_warmup_steps = int(warmup_pct * total_steps)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda, -1)

    def model_size(self):
        return sum(p.numel() for p in self.model.parameters())

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
        eval_results = {}

        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, **kw
            )

            batch_loader = self.get_batch_loader(train_data)

            total_steps = 1 + (len(batch_loader) // gca_batches) * num_epochs

            scheduler = self.get_lr_scheduler(optimizer, total_steps=total_steps, warmup_pct=warmup_pct)

            # === START OF TRAINING LOOP ===
            for epoch in range(start_epoch, start_epoch + num_epochs):
                log.info(f"Epoch #{epoch:02}")
                epoch_start = time.time()

                self.model.train()

                if use_tensorboard:
                    # write intial values
                    for i, lr in enumerate(scheduler.get_last_lr()):
                        tensorboard_writer.add_scalar(f"learning_rate/{i}", lr, global_step)
                    if epoch == start_epoch:
                        tensorboard_writer.add_scalar("epochs_completed", epoch, global_step)

                accumulated_batch_loss = 0
                epoch_loss = 0

                optimizer.zero_grad()
                for batch_index, batch in enumerate(batch_loader, start=1):
                    loss = self.forward_loss(batch)

                    epoch_loss += loss.detach().float()
                    accumulated_batch_loss += loss.detach().float()

                    loss = loss / gca_batches
                    loss.backward()

                    if batch_index % gca_batches == 0:
                        global_step += 1

                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        optimizer.zero_grad()

                        if use_tensorboard:
                            tensorboard_writer.add_scalar(
                                "loss/train_loss_batch", accumulated_batch_loss / gca_batches, global_step
                            )
                            for i, lr in enumerate(scheduler.get_last_lr()):
                                tensorboard_writer.add_scalar(f"learning_rate/{i}", lr, global_step)

                        accumulated_batch_loss = 0

                loss = epoch_loss / batch_index
                epoch_end = time.time()

                if use_tensorboard:

                    tensorboard_writer.add_scalar("epochs_completed", epoch + 1, global_step)

                    tensorboard_writer.add_scalar("loss/train_epoch_epoch", loss, global_step)
                    tensorboard_writer.add_scalar("model size/current", self.model_size(), global_step)

                    tensorboard_writer.add_scalar("time/epoch", epoch_end - epoch_start, global_step)

                log.info(f"Train loss: {loss:.4}")
                if test_data is not None:
                    eval_results = self.evaluate(test_data, batch_size=batch_size, num_workers=num_workers)
                    self.track_evaluation(eval_results, global_step, tensorboard_writer=tensorboard_writer)

        except KeyboardInterrupt:
            log_line(log)

            if propagate_interrupt:
                log.warning("Exiting from training early.")
                raise KeyboardInterrupt

            if test_data is not None:
                log.info("Evaluating after early termination:")
                eval_results = self.evaluate(test_data)
                self.track_evaluation(eval_results, global_step, tensorboard_writer=tensorboard_writer)
            else:
                log.warning("Exiting from training early.")

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

    def prepare_batch(self, batch):
        return {k: v.to(growing_transformer.device) for k, v in batch.items()}

    def forward_loss(self, batch):
        batch = self.prepare_batch(batch)
        outputs = self.model(**batch)
        return outputs.loss

    def track_evaluation(self, eval_results: Mapping, global_step: int = None, tensorboard_writer=None):
        for k, v in eval_results.items():
            log.info(f"Eval {k}: {v:.4}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(f"eval/{k}", v, global_step)

    def evaluate(self, test_data, batch_size=None, num_workers=None) -> Dict:
        log.info(f"Evaluating model on {len(test_data)} samples.")
        self.model.eval()

        loss_sum = 0.0

        with torch.no_grad():
            batch_loader = self.get_batch_loader(
                test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )

            for batch in batch_loader:
                batch = self.prepare_batch(batch)

                labels = batch["labels"]

                outputs = self.model(**batch)
                loss_sum += outputs[0].detach().float()
                prediction_scores = outputs[1]

                predictions = prediction_scores.argmax(-1)

                # only include masked tokens in metrics
                masked_tokens = (labels >= 0).view(-1)

                self.metric.add_batch(
                    predictions=predictions.view(-1)[masked_tokens],
                    references=labels.view(-1)[masked_tokens],
                )

            eval_loss = loss_sum / len(batch_loader)

            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            eval_results = self.metric.compute()

            assert eval_results is not None

            return {"eval_loss": eval_loss, "perplexity": perplexity, **eval_results}
