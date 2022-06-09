import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    default_data_collator,
    glue_processors,
)

import growing_transformer
from growing_transformer import BaseTrainer as Trainer
from growing_transformer.data import load_glue_task


@dataclass
class FinetuneConfig:
    task: str = "cola"
    device: str = "cuda:0"
    lr: float = 1e-5
    epochs: int = 5
    scheduler: str = "linear"
    warmup_portion: float = 0.06
    gca_batches: int = 1
    load: Optional[str] = None
    batch: int = 32
    checkpoint_every: Optional[int] = None
    save: bool = True
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="base_config", node=FinetuneConfig)


@hydra.main(config_path="config", config_name="finetuning")
def main(cfg):
    log = logging.getLogger("finetune")

    test_dir = Path("predictions")
    test_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
    )

    if cfg.device is not None:
        log.info(f"Set device to {cfg.device}")
        growing_transformer.device = torch.device(cfg.device)

    torch.manual_seed(cfg.seed)

    log.info(f"Using device {growing_transformer.device}.")

    log.info(f"Fine-tuning on task {cfg.task}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset, metric, num_labels = load_glue_task(cfg.task, tokenizer)

    if cfg.load is not None:
        path = Path(get_original_cwd()) / cfg.load
        log.info(f"Loading model from '{path}'")
        model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    else:
        # an example model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    trainer = Trainer(
        model,
        metric=metric,
        data_collator=default_data_collator,
        is_regression=(num_labels == 1),
        is_masked=False,
    )

    tensorboard_writer = SummaryWriter(".")

    def predict_testset(trainer, global_step, epoch=None):
        if cfg.task == "mnli":
            test_sets = [dataset["test"]]
            test_names = [""]
        else:
            test_sets = [dataset["test_matched"], dataset["test_mismatched"]]
            test_names = ["_matched", "_mismatched"]

        for test_set, test_name in zip(test_sets, test_names):
            batch_loader = trainer.get_batch_loader(test_set, batch_size=32, shuffle=False, drop_last=False)

            complete_predictions = None
            complete_ids = None

            for batch in batch_loader:
                batch = trainer.prepare_batch(batch)
                outputs = trainer.model(**batch)

                if not trainer.is_regression:
                    predictions = outputs.logits.argmax(-1)
                else:
                    predictions = outputs.logits.squeeze()

                if complete_predictions is None:
                    complete_predictions = predictions.detach().cpu().numpy()
                    complete_ids = batch["idx"].detach().cpu().numpy()
                else:
                    complete_predictions = np.append(complete_predictions, predictions.detach().cpu().numpy(), axis=0)
                    complete_ids = np.append(complete_ids, batch["idx"].detach().cpu().numpy(), axis=0)

            processor = glue_processors[cfg.task]()

            label_list = processor.get_labels()

            if trainer.is_regression:
                complete_predictions = [tr for tr in complete_predictions]
            else:
                complete_predictions = [label_list[tr] for tr in complete_predictions]

            test_pred = pd.DataFrame({"index": complete_ids, "prediction": complete_predictions})
            test_pred.to_csv(str(test_dir / f"{cfg.task}{cfg.task}_{epoch}.tsv"), sep="\t", index=False)

    if cfg.task == "mnli":
        val_data = {
            "mnli_matched": dataset["validation_matched"],
            "mnli_mismatched": dataset["validation_mismatched"],
        }
    else:
        val_data = dataset["validation"]

    trainer.train(
        train_data=dataset["train"],
        test_data=val_data,
        max_lr=cfg.lr,
        gca_batches=cfg.gca_batches,
        batch_size=cfg.batch,
        num_epochs=cfg.epochs,
        lr_scheduler_type=cfg.scheduler,
        lr_scheduler_warmup_portion=cfg.warmup_portion,
        tensorboard_writer=tensorboard_writer,
        weight_decay=cfg.weight_decay,
        checkpoint_every=cfg.checkpoint_every,
        custom_eval=predict_testset,
    )

    tensorboard_writer.close()

    if cfg.save:
        model.save_pretrained("finetuned")


if __name__ == "__main__":
    main()
