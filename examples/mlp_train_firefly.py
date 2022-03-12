import logging
from typing import Any, Dict

import torch
from sandbox import SimpleModel, SineToyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from growing_transformer import Trainer, GrowingConfig
from growing_transformer.trainer.util import GridSearch, log_line

log = logging.getLogger("growing_transformer")


train_data = SineToyDataset(8000)
grow_data = SineToyDataset(2000)

grid = GridSearch(
    dict(
        seed=range(5),
        learning_rate=[0.01, 0.005, 0.001],
        use_onecycle=[True, False],
        num_novel=[4],
        split=[True],
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

    log_line(log)

    torch.manual_seed(hparams["seed"])

    model = SimpleModel(1, 1, 8, 2, 2, config=GrowingConfig(**hparams))

    run_name = f"run_{i:04}"
    tensorboard_writer = SummaryWriter(f"runs/firefly/{run_name}")

    batch_loader = DataLoader(train_data, batch_size=32)
    for example_x, example_y in batch_loader:
        tensorboard_writer.add_graph(model, example_x)
        break

    def grow_func(trainer, model, growth_phase):

        sizes = list()
        for m in model.growing_modules():
            sizes.append(m.grow())

        trainer.tune_direction(grow_data)
        trainer.tune_new_parts(grow_data)

        trainer.calculate_new_gradient(grow_data)

        for m in model.growing_modules():
            selected = m.select(hparams["num_kept_neurons"])
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
