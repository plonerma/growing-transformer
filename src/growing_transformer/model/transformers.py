from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertForMaskedLM,
    BertOnlyMLMHead,
    BertPooler,
    MaskedLMOutput,
)

from ..configuration import GrowingConfig
from .base import Growing, truncated_normal_
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

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
    ):
        embeded = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
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

        # tie input and output embeddings
        self.cls.predictions.decoder.weight = self.bert.get_input_embeddings().weight

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
        labels: Tensor = None,
        return_dict: Optional[bool] = None,
        **kw,
    ) -> Union[Tuple[Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = outputs
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # -100 index = padding token
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )

    def save_pretrained(self, **kwargs):
        dummy = BertForMaskedLM(self.config)
        return dummy.save_pretrained(state_dict=self.state_dict(), **kwargs)


class HuggingfaceMLMTransformer(BertForMaskedLM):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
