import logging
import math
import time
from typing import Dict, List, Optional, Tuple

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
        custom_eval=None,
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
        self.custom_eval = custom_eval

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

    def get_lr_scheduler(
        self, *, optimizer, type: str, warmup_steps: int = None, total_steps: int = None, last_step: int = None
    ):
        if last_step is None:
            last_step = -1

        if type == "constant":
            return LambdaLR(optimizer, lambda step: 1, last_step)
        elif type == "linear":
            assert warmup_steps is not None, "Linear lr scheduler needs number of warmup steps."
            assert total_steps is not None, "Linear lr scheduler needs number of total steps to plan for."

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

            return LambdaLR(optimizer, lr_lambda, last_step)
        else:
            raise KeyError(f"Scheduler {type} not implemented.")

    def model_size(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(
        self,
        train_data,
        test_data=None,
        max_lr: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-06,
        weight_decay: float = 0.01,
        gca_batches: int = 16,  # gradient accumulation batches
        num_epochs: int = 5,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        propagate_interrupt=False,
        tensorboard_writer=None,
        log_training_info=True,
        start_epoch: int = 0,
        global_step: int = 0,
        quit_on_step: int = None,
        no_decay: List[str] = ["bias", "layer_norm.weight", "LayerNorm.weight"],
        lr_scheduler_type: str = None,
        lr_scheduler_warmup_steps: int = None,
        lr_scheduler_warmup_portion: float = None,
        lr_scheduler_num_epochs: int = None,
        lr_scheduler_last_step: int = None,
        **kw,
    ):

        self.model.to(growing_transformer.device)

        if log_training_info:
            log.info(f"Model: {self.model}")
            log_line(log)

        use_tensorboard = tensorboard_writer is not None
        eval_results = {}

        try:
            if lr_scheduler_type is None:
                lr_scheduler_type = "constant"

            # separate parameters into those with and without weight decay
            param_groups = [
                # parameters with weight decay
                {
                    "params": [
                        p
                        for name, p in self.model.named_parameters()
                        if not any(name.endswith(suffix) for suffix in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "initial_lr": max_lr,
                },
                # parameters without weight decay
                {
                    "params": [
                        p
                        for name, p in self.model.named_parameters()
                        if any(name.endswith(suffix) for suffix in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "initial_lr": max_lr,
                },
            ]

            optimizer = torch.optim.AdamW(
                param_groups,
                betas=betas,
                eps=eps,
                **kw,
            )

            scaler = torch.cuda.amp.GradScaler()

            batch_loader = self.get_batch_loader(train_data)

            if lr_scheduler_type is not None:

                if lr_scheduler_num_epochs is not None:
                    total_steps = 1 + (len(batch_loader) // gca_batches) * lr_scheduler_num_epochs
                else:
                    total_steps = 1 + (len(batch_loader) // gca_batches) * num_epochs

                warmup_steps: Optional[int]
                if lr_scheduler_warmup_steps is None and lr_scheduler_warmup_portion is not None:
                    warmup_steps = int(total_steps * lr_scheduler_warmup_portion)
                    warmup_info = f"{lr_scheduler_warmup_portion:.1%} ({warmup_steps} steps) warmup"

                else:
                    warmup_steps = lr_scheduler_warmup_steps
                    warmup_info = f"{warmup_steps} steps warmup"

                log.info(
                    f"Scheduler: {lr_scheduler_type}, {total_steps} steps total, {warmup_info}, last step: {lr_scheduler_last_step}"
                )

                scheduler = self.get_lr_scheduler(
                    optimizer=optimizer,
                    type=lr_scheduler_type,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    last_step=lr_scheduler_last_step,
                )
            else:
                scheduler = None

            log_line(log)
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

                accumulated_batch_loss = 0.0

                optimizer.zero_grad()
                for batch_index, batch in enumerate(batch_loader, start=1):

                    with torch.cuda.amp.autocast():
                        loss = self.forward_loss(batch)
                        loss = loss / gca_batches

                        accumulated_batch_loss += loss.detach().float()

                    scaler.scale(loss).backward()

                    if (batch_index % gca_batches == 0) or (batch_index == len(batch_loader)):
                        global_step += 1

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        if scheduler is not None:
                            scheduler.step()

                        if use_tensorboard:
                            if not (batch_index % gca_batches == 0):
                                # make correction to average loss
                                accumulated_batch_loss *= gca_batches / (batch_index % gca_batches)

                            tensorboard_writer.add_scalar("loss/train_loss_batch", accumulated_batch_loss, global_step)
                            for i, lr in enumerate(scheduler.get_last_lr()):
                                tensorboard_writer.add_scalar(f"learning_rate/{i}", lr, global_step)

                        accumulated_batch_loss = 0.0

                        if quit_on_step is not None and global_step == quit_on_step:
                            raise KeyboardInterrupt

                epoch_end = time.time()

                if use_tensorboard:

                    tensorboard_writer.add_scalar("epochs_completed", epoch + 1, global_step)
                    tensorboard_writer.add_scalar("model size/current", self.model_size(), global_step)
                    tensorboard_writer.add_scalar("time/epoch", epoch_end - epoch_start, global_step)

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
            epoch=epoch,
            total_steps=(num_epochs * len(batch_loader)) // gca_batches,
            global_step=global_step,
            **eval_results,
        )

        return results

    def prepare_batch(self, batch):
        return {k: v.to(growing_transformer.device) for k, v in batch.items()}

    def forward_loss(self, batch):
        batch = self.prepare_batch(batch)
        outputs = self.model(**batch)
        return outputs.loss

    def track_evaluation(self, eval_results: Dict, global_step: int = None, tensorboard_writer=None):
        with torch.no_grad():
            if self.custom_eval is not None:
                r = self.custom_eval(self, global_step=global_step)
                if r is not None:
                    assert isinstance(r, dict)
                    eval_results.update(r)

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
