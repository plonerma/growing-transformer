import logging
from typing import Any, Dict

import torch
from sandbox import SimpleModel, SineToyDataset
from torch.utils.tensorboard import SummaryWriter

from growing_transformer import Trainer
from growing_transformer.train_util import GridSearch, log_line

log = logging.getLogger("growing_transformer")


train_data = SineToyDataset(8000)
grow_data = SineToyDataset(2000)

grid = GridSearch(
    dict(
        seed=range(1),
        learning_rate=[0.01, 0.005, 0.001],
        use_onecycle=[True, False],
        tune_direction=[True, False],
        tune_new_neurons=[True, False],
        neuron_selection=["random", "firefly"],
        num_novel=[0, 4],
        split=[False, True],
    )
)

log.info(f"Searching grid with {len(grid)} elements.")

for i, p in enumerate(grid):
    hparams: Dict[str, Any] = dict(
        learning_rate=0.005,
        growth_phases=5,
        num_epochs=5,
        use_onecycle=True,
        num_kept_neurons=2,
    )

    hparams.update(p)

    if (hparams["num_novel"] == 0) and not hparams["split"]:
        continue

    log_line(log)
    log.info("Hyperparameters")
    for k, v in hparams.items():
        log.info(f" - {k}: {str(v)}")

    log_line(log)

    torch.manual_seed(hparams["seed"])

    model = SimpleModel(1, 1, 8, 2, 2, config=hparams)

    run_name = f"run_{i:04}"
    tensorboard_writer = SummaryWriter(f"runs/variations/{run_name}")

    def grow_func(trainer, model, growth_phase):

        sizes = list()
        for m in model.growing_modules():
            sizes.append(m.grow())

        if hparams["tune_direction"]:
            trainer.tune_direction(grow_data)

        if hparams["tune_new_neurons"]:
            trainer.tune_new_neurons(grow_data)

        if hparams["neuron_selection"] == "firefly":
            trainer.calculate_new_gradient(grow_data)

            for m in model.growing_modules():
                selected = m.select(hparams["num_kept_neurons"])
                m.degrow(selected)
                if selected.numel():
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, growth_phase)
        else:
            assert hparams["neuron_selection"] == "random"

            for s, m in zip(sizes, model.growing_modules()):

                if len(s) == 0:
                    continue
                *shape, n_neurons = s

                selected = torch.stack(
                    [torch.randperm(n_neurons)[: hparams["num_kept_neurons"]] for _ in range(s.numel() // n_neurons)]
                )

                selected = selected.reshape(*shape, -1)

                m.degrow(selected)
                if selected.numel():
                    tensorboard_writer.add_histogram(f"selected neurons/{m.__class__.__name__}", selected, growth_phase)

    trainer = Trainer(model, grow_func)

    try:
        metrics = trainer.train(train_data, propagate_interrupt=True, tensorboard_writer=tensorboard_writer, **hparams)
        tensorboard_writer.add_hparams(hparams, metrics, run_name=".")
    except KeyboardInterrupt:
        log_line(log)
        log.warning("Quitting grid search.")
        break
    finally:
        tensorboard_writer.close()
