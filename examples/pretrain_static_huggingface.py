import logging
from pathlib import Path

import datasets
import torch
from datasets import DatasetDict
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

import growing_transformer
from growing_transformer import BaseTrainer
from growing_transformer.configuration import GrowingConfigFull
from growing_transformer.data import MLMSegmenetDataset
from growing_transformer.model import HuggingfaceMLMTransformer
from growing_transformer.trainer.util import add_file_handler

growing_transformer.device = torch.device("cuda:0")

base_path = Path("results/pretrained_static_huggingface_small_01")
base_path.mkdir(exist_ok=True, parents=True)

log = logging.getLogger("growing_transformer")

log_handler = add_file_handler(log, base_path / "training.log")


corpus = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")

assert isinstance(corpus, DatasetDict)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

train_data = MLMSegmenetDataset(corpus["train"], tokenizer)  # .downsampled(0.1)
test_data = MLMSegmenetDataset(corpus["test"], tokenizer)

config = GrowingConfigFull()

# model = GrowingMLMTransformer(config)
model = HuggingfaceMLMTransformer(config)

tensorboard_writer = SummaryWriter(base_path / "run")

trainer = BaseTrainer(model)


trainer.train(
    train_data, num_epochs=24, test_data=test_data, tensorboard_writer=tensorboard_writer, batch_size=64, gca_batches=4
)

torch.save(model.state_dict(), base_path / "trained_model.pt")
