import datasets
from datasets import DatasetDict
from transformers import BertTokenizer

from growing_transformer import (
    BaseTrainer,
    GrowingConfig,
    GrowingMLMTransformer,
    GrowthSchedule,
)
from growing_transformer.data import MLMSegmenetDataset

corpus = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

assert isinstance(corpus, DatasetDict)

tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/bert-base-cased")

train_data = MLMSegmenetDataset(corpus["train"], tokenizer)

config = GrowingConfig()

model = GrowingMLMTransformer(config)

trainer = BaseTrainer(model)

trainer.train(train_data, schedule=GrowthSchedule(10))
