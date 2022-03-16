import datasets
from datasets import DatasetDict
from transformers import BertTokenizer

from growing_transformer import (
    GrowingConfig,
    GrowingMLMTransformer,
    GrowingTrainer,
    GrowthSchedule,
)
from growing_transformer.data import MLMSegmenetDataset
from growing_transformer.model import GrowingMLP as MLP

corpus = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

assert isinstance(corpus, DatasetDict)

tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/bert-base-cased")

train_data = MLMSegmenetDataset(corpus["train"], tokenizer)

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
