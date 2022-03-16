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
from growing_transformer.trainer.util import subsample

corpus = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

assert isinstance(corpus, DatasetDict)

tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/bert-base-cased")

train_data = MLMSegmenetDataset(corpus["train"], tokenizer)
train_data = subsample(train_data, 0.1)
test_data = MLMSegmenetDataset(corpus["test"], tokenizer)

config = GrowingConfig()

model = GrowingMLMTransformer(config)

trainer = BaseTrainer(model)

trainer.train(train_data, schedule=GrowthSchedule(10), test_data=test_data)

torch.save(model.state_dict(), "trained_model.pt")
