from typing import Mapping, Optional

import torch
from torch import BoolTensor, Tensor
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertOnlyMLMHead,
    BertPooler,
)

from ..configuration import GrowingConfig
from .base import Growing
from .encoder import GrowingEncoder


class GrowingTransformer(Growing):
    def __init__(self, config: GrowingConfig, add_pooling_layer=False):
        super().__init__(config=config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = GrowingEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids: Tensor, attention_mask: Optional[BoolTensor] = None):

        embeded = self.embeddings(
            input_ids=input_ids,
        )

        encoded = self.encoder(embeded, attention_mask=attention_mask)

        if self.pooler is not None:
            return self.pooler(encoded)
        else:
            return encoded


class GrowingMLMTransformer(Growing):
    """
    Growing Transformer Model with a `masked language modeling` head for pretraining the transformer.
    """

    def __init__(self, config):
        super().__init__(config)

        self.bert = GrowingTransformer(config)
        self.cls = BertOnlyMLMHead(config)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, inputs: Tensor, attention_mask: Optional[BoolTensor] = None, mlm_mask: Optional[BoolTensor] = None
    ):
        return self.cls(self.bert(inputs, attention_mask=attention_mask))

    def forward_loss(self, batch: Mapping[str, Tensor]):
        prediction_scores = self(batch["input_masked"], attention_mask=batch["attention_mask"])

        num_classes = prediction_scores.size(-1)

        prediction_scores = torch.masked_select(prediction_scores, batch["mlm_mask"][..., None]).view(-1, num_classes)

        # select relevant labels
        labels = torch.masked_select(batch["input_ids"], batch["mlm_mask"])

        masked_lm_loss = self.criterion(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return masked_lm_loss

    def evaluate(self, data, batch_size=32, num_workers=None):
        batch_loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if num_workers is None else num_workers,
        )

        total_samples = 0
        correct = 0
        loss = 0.0

        for batch in batch_loader:
            prediction_scores = self(batch["input_masked"], attention_mask=batch["attention_mask"])
            num_classes = prediction_scores.size(-1)

            prediction_scores = torch.masked_select(prediction_scores, batch["mlm_mask"][..., None]).view(
                -1, num_classes
            )
            predicted = prediction_scores.argmax(-1)

            # select relevant labels
            labels = torch.masked_select(batch["input_ids"], batch["mlm_mask"])

            loss += self.criterion(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            correct += (predicted == labels).sum()

            total_samples += labels.size(0)

        return dict(
            accuracy=correct / total_samples,
            eval_loss=loss / total_samples,
        )
