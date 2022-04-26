import math
from typing import Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

import growing_transformer


class HuggingfaceMLMTransformer(BertForMaskedLM):
    def forward_loss(self, batch: Mapping[str, Tensor]):
        assert (
            self.config.loss_on_all_tokens
        ), "Huggingface transformer only supports calculation of loss on all tokens."

        input_ids = batch["input_ids"].to(growing_transformer.device)
        attention_mask = batch["attention_mask"].to(growing_transformer.device)
        labels = batch["labels"].to(growing_transformer.device)

        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs[0]

    def evaluate(self, batch_loader, batch_size=32, num_workers=None):
        total_samples = 0
        total_masked_samples = 0
        correct = 0
        loss = 0.0

        for batch in batch_loader:
            input_ids = batch["input_ids"].to(growing_transformer.device)
            attention_mask = batch["attention_mask"].to(growing_transformer.device)
            labels = batch["labels"].to(growing_transformer.device)

            outputs = self(input_ids, attention_mask=attention_mask, labels=labels)

            loss, prediction_scores = outputs[:2]

            num_classes = prediction_scores.size(-1)

            mlm_mask = labels >= 0

            predicted = torch.masked_select(prediction_scores, mlm_mask[..., None]).view(-1, num_classes).argmax(-1)

            # select relevant labels
            labels = torch.masked_select(input_ids, mlm_mask)

            correct += (predicted == labels).sum()

            total_masked_samples += labels.size(0)
            total_samples += (input_ids >= 0).sum()

        eval_loss = loss / len(batch_loader)

        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return dict(
            accuracy=correct / total_masked_samples,
            eval_loss=eval_loss,
            perplexity=perplexity,
        )
