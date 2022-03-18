import logging
from pathlib import Path

import datasets
import torch
from datasets import DatasetDict
from transformers import BertTokenizer

from growing_transformer import (
    BaseTrainer,
    GrowingConfig,
    GrowingMLMTransformer,
    GrowthSchedule,
)


from growing_transformer.data import MLMSegmenetDataset
from growing_transformer.trainer.util import add_file_handler


base_path = Path("results/pretrained_growing")
base_path.mkdir(exist_ok=True, parents=True)

log = logging.getLogger("growing_transformer")

log_handler = add_file_handler(log, base_path / "training.log")


corpus = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

assert isinstance(corpus, DatasetDict)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

train_data = MLMSegmenetDataset(corpus["train"], tokenizer).downsampled(0.1)
test_data = MLMSegmenetDataset(corpus["test"], tokenizer)

config = GrowingConfig()

model = GrowingMLMTransformer(config)

trainer = GrowingTrainer(model)

schedule = GrowthSchedule(2)

schedule.add_phase(
    epochs=2,
    grow={MLP: dict(split=True, num_novel=16)},
    num_new_parts={MLP: 16},
)

schedule.add_phase(
    epochs=2,
    grow={MLP: dict(split=True, num_novel=16)},
    num_new_parts={MLP: 16},
)

schedule.add_phase(
    epochs=2,
    grow={MLP: dict(split=True, num_novel=16)},
    num_new_parts={MLP: 16},
)

schedule.add_phase(
    epochs=2,
    grow={MLP: dict(split=True, num_novel=16)},
    num_new_parts={MLP: 16},
)

trainer.train(train_data, schedule=schedule)
torch.save(model.state_dict(), base_path / "trained_model.pt")
