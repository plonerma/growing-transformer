import logging
from typing import Any, Dict

import torch
from sandbox import SimpleModel, SineToyDataset
from torch.utils.tensorboard import SummaryWriter

from growing_transformer import GrowingConfig, GrowingTrainer, GrowthSchedule
from growing_transformer.model import GrowingMLP as MLP
from growing_transformer.trainer.util import GridSearch, log_line

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

    model = SimpleModel(1, 1, 8, 2, 2, config=GrowingConfig(**hparams))

    run_name = f"run_{i:04}"
    tensorboard_writer = SummaryWriter(f"runs/variations/{run_name}")

    trainer = GrowingTrainer(model, tune_direction=True, tune_new_parts=True, selection_method="firefly")

    schedule = GrowthSchedule(hparams["num_epochs"])

    for _ in range(hparams["growth_phases"]):
        schedule.add_phase(
            epochs=hparams["num_epochs"],
            grow={MLP: dict(split=hparams["split"], num_novel=hparams["num_novel"])},
            num_new_parts={MLP: hparams["num_new_parts"]},
        )

    try:
        metrics = trainer.train(
            train_data, schedule=schedule, propagate_interrupt=True, tensorboard_writer=tensorboard_writer, **hparams
        )
        tensorboard_writer.add_hparams(hparams, metrics, run_name=".")
    except KeyboardInterrupt:
        log_line(log)
        log.warning("Quitting grid search.")
        break
    finally:
        tensorboard_writer.close()
