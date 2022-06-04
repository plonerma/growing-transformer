import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    default_data_collator,
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
    )

    tensorboard_writer = SummaryWriter(".")

    trainer.train(
        train_data=dataset["train"],
        test_data=dataset["validation"],
        max_lr=cfg.lr,
        gca_batches=cfg.gca_batches,
        batch_size=16,
        num_epochs=cfg.epochs,
        lr_scheduler_type=cfg.scheduler,
        lr_scheduler_warmup_portion=cfg.warmup_portion,
        tensorboard_writer=tensorboard_writer,
        weight_decay=cfg.weight_decay,
        checkpoint_every=cfg.checkpoint_every,
    )

    tensorboard_writer.close()

    if cfg.save:
        model.save_pretrained("finetuned")


if __name__ == "__main__":
    main()
