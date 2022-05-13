import argparse
import logging
from datetime import datetime
from pathlib import Path

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
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--epochs", type=float, help="number of epochs", default=10)
    parser.add_argument("--warmup_pct", type=float, help="warmup time (as float < 1.0)", default=0.06)
    parser.add_argument("--gca", type=float, help="number of batches to accumulate for each update step", default=2)
    parser.add_argument("--load_model", type=Path, help="where to loade the model from", default=None)
    parser.add_argument("--save_model", type=Path, help="where to save the fine tuned model", default=None)
    parser.add_argument(
        "--batch", type=float, help="batch size (multiple batches may be accumulated using --gca)", default=16
    )
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

    if args.load_model is not None:
        model = BertForSequenceClassification.from_pretrained(args.load_model, num_labels=num_labels)
    else:
        # an example model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    trainer = Trainer(
        model,
        metric=metric,
        data_collator=default_data_collator,
    )

    path = datetime.now().strftime("outputs/%Y-%m-%d/%H-%M-%S")

    tensorboard_writer = SummaryWriter(path)

    trainer.train(
        train_data=dataset["train"],
        test_data=dataset["validation"],
        max_lr=args.lr,
        gca_batches=args.gca,
        batch_size=16,
        num_epochs=args.epochs,
        warmup_pct=args.warmup_pct,
        tensorboard_writer=tensorboard_writer,
    )

    if args.save_model is not None:
        torch.save(model.state_dict(), parser.save_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
