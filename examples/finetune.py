import argparse
import logging

from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    default_data_collator,
)

import growing_transformer
from growing_transformer import BaseTrainer as Trainer
from growing_transformer.data import load_glue_task

log = logging.getLogger("fine-tuning")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model.")
    parser.add_argument("task", metavar="TASK", type=str, help="task to finetune on")
    parser.add_argument("--device", type=str, help="device to use for finetuning", default=None)
    return parser.parse_args()


def main(args):

    logging.basicConfig(level=logging.INFO)

    if args.device is not None:
        log.info(f"Set device to {args.device}")
        growing_transformer.device = torch.device(args.device)

    log.info(f"Using device {growing_transformer.device}.")

    log.info(f"Fine-tuning on task {args.task}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset, metric, num_labels = load_glue_task(args.task, tokenizer)

    # an example model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    trainer = Trainer(
        model,
        data_collator=default_data_collator,
    )

    path = datetime.now().strftime("outputs/%Y-%m-%d/%H-%M-%S")

    tensorboard_writer = SummaryWriter(path)

    trainer.train(
        train_data=dataset["train"],
        test_data=dataset["validation"],
        max_lr=1e-6,
        gca_batches=2,
        batch_size=16,
        num_epochs=4,
        tensorboard_writer=tensorboard_writer,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
