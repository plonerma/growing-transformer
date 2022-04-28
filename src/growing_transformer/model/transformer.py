from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertOnlyMLMHead,
    BertPooler,
    MaskedLMOutput,
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
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
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
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
