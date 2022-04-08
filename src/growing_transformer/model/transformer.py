import math
from typing import Mapping, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertOnlyMLMHead,
    BertPooler,
)

import growing_transformer

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

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None):

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

    def forward(self, inputs: Tensor, attention_mask: Optional[Tensor] = None, mlm_mask: Optional[Tensor] = None):
        inputs.to(growing_transformer.device)
        return self.cls(self.bert(inputs, attention_mask=attention_mask))

    def forward_loss(self, batch: Mapping[str, Tensor]):
        masked_input = batch["input_masked"].to(growing_transformer.device)
        attention_mask = batch["attention_mask"].to(growing_transformer.device)
        mlm_mask = batch["mlm_mask"].to(growing_transformer.device)
        input_ids = batch["input_ids"].to(growing_transformer.device)

        prediction_scores = self(masked_input, attention_mask=attention_mask)

        num_classes = prediction_scores.size(-1)

        if not self.config.loss_on_all_tokens:
            prediction_scores = torch.masked_select(prediction_scores, mlm_mask[..., None]).view(-1, num_classes)

            # select relevant labels
            labels = torch.masked_select(input_ids, mlm_mask)

        else:
            # for consistency with huggingface MLM transformr, calculate loss on
            # all token
            prediction_scores = prediction_scores.view(-1, num_classes)
            labels = input_ids

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
            masked_input = batch["input_masked"].to(growing_transformer.device)
            attention_mask = batch["attention_mask"].to(growing_transformer.device)
            mlm_mask = batch["mlm_mask"].to(growing_transformer.device)
            input_ids = batch["input_ids"].to(growing_transformer.device)

            prediction_scores = self(masked_input, attention_mask)
            num_classes = prediction_scores.size(-1)

            prediction_scores = torch.masked_select(prediction_scores, mlm_mask[..., None]).view(-1, num_classes)
            predicted = prediction_scores.argmax(-1)

            # select relevant labels
            labels = torch.masked_select(input_ids, mlm_mask)

            if not self.config.loss_on_all_tokens:
                prediction_scores = torch.masked_select(prediction_scores, mlm_mask[..., None]).view(-1, num_classes)

                # select relevant labels
                labels = torch.masked_select(input_ids, mlm_mask)
                loss += self.criterion(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            else:
                # for consistency with huggingface MLM transformr, calculate loss on
                # all token
                prediction_scores = prediction_scores.view(-1, num_classes)

                loss += self.criterion(prediction_scores.view(-1, self.config.vocab_size), input_ids.view(-1))

            correct += (predicted == labels).sum()

            total_samples += labels.size(0)

        eval_loss = loss / len(batch_loader)

        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return dict(
            accuracy=correct / total_samples,
            eval_loss=eval_loss,
            perplexity=perplexity,
        )
