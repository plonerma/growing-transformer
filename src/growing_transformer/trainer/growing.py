import logging
import re
import time
from contextlib import contextmanager
from typing import Mapping, Optional

import torch

import growing_transformer

from ..data import downsample_dataset
from ..model import GrowingModule
from .base import BaseTrainer
from .schedule import GrowthSchedule
from .util import log_line

log = logging.getLogger("growing_transformer")


class GrowingTrainer(BaseTrainer):
    def __init__(
        self, model, *, tune_direction: bool = True, tune_step_size: bool = True, selection_method="firefly", **kws
    ):
        super().__init__(model, **kws)

        self._tune_direction = tune_direction
        self._tune_step_size = tune_step_size
        self._selection_method = selection_method

    def grow_model(
        self,
        substeps,
        grow_data=None,
        tensorboard_writer=None,
        index=None,
        batch_size: int = 32,
        gca_batches: int = 1,
        use_onecycle: int = 1,
        num_epochs: int = 1,
        shuffle=True,
        num_workers: Optional[int] = None,
        track_tuned_steps=True,
        **optimizer_params,
    ):
        log.info("Growing model")

        # === Match and grow modules ===
        grown_modules = list()

        start_size = self.model_size()
        log.info(f"Model size currently: {start_size}")

        for conf in substeps:
            if conf.get("match_end", True):
                pattern_str = conf["match"] + "$"
            else:
                pattern_str = conf["match"]

            pattern = re.compile(pattern_str)

            # configuration used for growing
            num_novel = conf["num_novel"]
            split = conf["split"]

            num_matched = 0
            for name, m in self.model.growing_modules(named=True):
                if re.search(pattern, name) is not None and isinstance(m, GrowingModule):
                    size = m.grow(num_novel=num_novel, split=split)
                    grown_modules.append((m, size, conf))

                    # already update config, so that it will be correct for newly added modules (eg. layers)
                    m.update_config(conf["num_keep"])

                    num_matched += 1

            log.info(f"Matched {num_matched} to {pattern_str} ({conf}).")

        self.model.to(growing_transformer.device)

        overgrown_size = self.model_size()
        log.info(f"Model size (overgrown): {overgrown_size}")

        # === Tune direction and step sizes ===
        if self._tune_direction or self._tune_step_size:
            assert grow_data is not None

            tuned_str = []
            if self._tune_direction:
                tuned_str.append("direction")
            if self._tune_step_size:
                tuned_str.append("step_size")

            log.info(f"Tuning {' and '.join(tuned_str)}")
            log.info(f"- gca_batches: {gca_batches}")
            log.info(f"- batch_size: {batch_size}")
            log.info(f"- num_epochs: {num_epochs}")
            log.info(f"- use_onecycle: {use_onecycle}")

            param_groups = list()
            relevant_params = list()

            for m, size, conf in grown_modules:
                kw = conf.get("tune_params", dict())

                if self._tune_direction:
                    params = [p for p in m._direction_params() if p is not None]
                    param_groups.append({"params": params, **kw.get("direction", dict())})
                    relevant_params += params
                if self._tune_step_size:
                    params = [m.step_size]
                    param_groups.append({"params": params, **kw.get("step_size", dict())})
                    relevant_params += params

            kw = dict(optimizer_params)
            lr = kw.pop("learning_rate")
            optimizer = torch.optim.AdamW(param_groups, lr=lr, **kw)

            total_steps = num_epochs * len(grow_data) // (batch_size * gca_batches) + 1

            scheduler = self.get_lr_scheduler(optimizer, total_steps=total_steps, warmup_pct=0.1)

            log.info(f"Tuning for {num_epochs} epochs.")

            with self.some_grad_only(*relevant_params):
                for epoch in range(num_epochs):
                    batch_loader = self.get_batch_loader(grow_data)

                    optimizer.zero_grad()

                    step_loss = 0.0

                    for batch_index, batch in enumerate(batch_loader, start=1):
                        loss = self.forward_loss(batch)
                        loss.backward()
                        step_loss += loss.detach()

                        if batch_index % gca_batches == 0:

                            tune_step = epoch * len(batch_loader) // gca_batches + batch_index // gca_batches
                            step_loss = step_loss / gca_batches

                            if tensorboard_writer is not None:
                                tensorboard_writer.add_scalar(
                                    f"tuning directions and steps/step {index}", step_loss, tune_step
                                )

                                if track_tuned_steps:
                                    for n, m in self.model.growing_modules(named=True):
                                        if isinstance(m, GrowingModule) and m.step_size is not None:
                                            tensorboard_writer.add_histogram(
                                                f"step_sizes/step {index}/{n}", m.step_size, tune_step
                                            )

                                        if isinstance(m, GrowingModule) and m.step_size is not None:
                                            tensorboard_writer.add_histogram(
                                                f"step_sizes/grad {index}/{n}", m.step_size.grad, tune_step
                                            )

                            optimizer.step()

                            if scheduler is not None:
                                scheduler.step()

                            optimizer.zero_grad()
                            step_loss = 0.0

        log.info(f"Selecting neurons to keep using {self._selection_method }")

        # === Select neurons ===
        if self._selection_method == "firefly":
            self.calculate_step_gradient(grow_data, batch_size=256)

            for m, size, conf in grown_modules:
                num_kept_parts = conf["num_keep"]
                selected = m.select(num_kept_parts)
                m.degrow(selected)
                if tensorboard_writer is not None and selected.numel() and index is not None:
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, index)

        elif self._selection_method == "random":
            for m, size, conf in grown_modules:

                num_kept_parts = conf["num_keep"]

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

        grown_size = self.model_size()
        log.info(f"Model size (after selection): {grown_size}")

        if tensorboard_writer:
            tensorboard_writer.add_scalar("model size/start size", start_size, index)
            tensorboard_writer.add_scalar("model size/overgrown size", overgrown_size, index)
            tensorboard_writer.add_scalar("model size/grown size", grown_size, index)
            tensorboard_writer.add_scalar("model size/overgrown ratio", overgrown_size / start_size, index)
            tensorboard_writer.add_scalar("model size/growth ratio", grown_size / start_size, index)

    def train(  # type: ignore[override]
        self,
        train_data,
        schedule: GrowthSchedule,
        grow_data_portion=1.0,
        test_data=None,
        grow_data=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        propagate_interrupt=False,
        tensorboard_writer=None,
        log_training_info=True,
        grow_tune_params: Mapping = {},
        **kw,
    ):
        if log_training_info:

            log.info(f"Model: {self.model}")
            log_line(log)
            log.info(f" - growth phases: {len(schedule)}")
            log.info(f" - number of samples: {len(train_data)}")
            log_line(log)

        global_step = 0
        current_epoch = 0
        train_info = None

        self.model.to(growing_transformer.device)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("model size/current", self.model_size(), global_step)

        for step_index, (step_type, step_params) in enumerate(schedule):
            if step_type == step_type.grow:
                if grow_data is None:
                    _grow_data = train_data
                else:
                    _grow_data = grow_data

                if grow_data_portion is not None and grow_data_portion < 1.0:
                    log.info("Downsampling grow data")
                    _grow_data = downsample_dataset(_grow_data, grow_data_portion)

                log.info(f"{len(_grow_data)} samples used for tuning growth.")

                grow_start = time.time()
                self.grow_model(
                    step_params,
                    grow_data=_grow_data,
                    tensorboard_writer=tensorboard_writer,
                    index=step_index,
                    num_workers=num_workers,
                    **grow_tune_params,
                )
                grow_end = time.time()

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("time/growth", grow_end - grow_start, global_step)

                    tensorboard_writer.add_scalar("model size/current", self.model_size(), global_step)

            elif step_type == step_type.train:
                train_params = {**kw, **step_params}

                train_start = time.time()

                train_info = super().train(
                    train_data=train_data,
                    test_data=test_data,
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

                current_epoch += 1

            elif step_type == step_type.checkpoint:
                torch.save(self.model.state_dict(), f"checkpoints/checkpoint_{step_index}.pt")

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

    def calculate_step_gradient(
        self, train_data, batch_size: int = 32, shuffle: bool = True, num_workers: Optional[int] = None
    ):
        with self.some_grad_only(*self.model.step_size_params()):
            self.model.zero_grad()

            for batch in self.get_batch_loader(train_data):
                loss = self.forward_loss(batch)
                loss.backward()
