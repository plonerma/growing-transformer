from transformers import BertForSequenceClassification, BertTokenizer

from growing_transformer import BaseTrainer as Trainer
from growing_transformer.data.glue import load_glue_dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = load_glue_dataset("cola", tokenizer)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

trainer = Trainer(model)

trainer.train(
    train_data=dataset["train"],
    test_data=dataset["test"],
    max_lr=1e-6,
    gca_batches=1,
    batch_size=32,
    num_epochs=4,
)
