import logging

import datasets
import pytest
import torch
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DataCollatorForLanguageModeling,
)

from growing_transformer import GrowingConfig, GrowingMLMTransformer, GrowingTransformer
from growing_transformer.data import prepare_mlm_dataset

log = logging.getLogger("growing_transformer.tests")


class TestGrowingTransformer:
    @pytest.mark.slow
    def test_mlm_transformer_function(self):
        # initialize growing transformer
        config = GrowingConfig()
        growing_model = GrowingMLMTransformer(config)

        # get state from that model
        state = growing_model.state_dict()

        # initialize bert transformer
        bert_model = BertForMaskedLM(config)

        # load state for growing model into bert model
        bert_model.load_state_dict(state)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        max_seq_length = 64

        raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

        assert "train" in raw_datasets, "Dataset does not contain train data."

        # remove headings
        def is_heading(text):
            text = text.strip()
            return text.startswith("=") and text.endswith("=")

        raw_datasets = raw_datasets.filter(lambda sample: not is_heading(sample["text"]))

        tokenized_datasets = prepare_mlm_dataset(
            tokenizer=tokenizer,
            dataset=raw_datasets,
            ignore_cache=False,
            max_seq_length=max_seq_length,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        data_loader = torch.utils.data.DataLoader(
            tokenized_datasets["train"], batch_size=1, shuffle=False, collate_fn=data_collator
        )

        for batch in data_loader:
            # make sure the models are in train mode
            # by setting the seed prior to computing the results, we can give
            # the same starting condition
            growing_model.train()
            bert_model.train()


            torch.manual_seed(42)
            growing_model.zero_grad()
            y_a = growing_model(**batch)
            y_a.loss.backward()

            torch.manual_seed(42)
            bert_model.zero_grad()
            y_b = bert_model(**batch)
            y_b.loss.backward()

            diff = torch.abs(y_a.logits - y_b.logits)

            log.info(f"Max. difference: {diff.max()}")

            assert torch.all(diff < 1e-12)

            # If gradient in lower level is the same, there should not be any
            # issue further up the chain
            pa = growing_model.bert.encoder.layer[0].attention.layer_norm.weight
            pb = bert_model.bert.encoder.layer[0].attention.output.LayerNorm.weight

            diff = torch.abs(pa.grad - pb.grad)

            log.info(f"Max. grad difference: {diff.max()}")

            assert torch.all(diff < 1e-12)

            break

    @pytest.mark.slow
    def test_transformer_model_function(self):
        # initialize growing transformer
        config = GrowingConfig()
        growing_model = GrowingTransformer(config, add_pooling_layer=False)

        # get state from that model
        state = growing_model.state_dict()

        # initialize bert transformer
        bert_model = BertModel(config, add_pooling_layer=False)

        # load state for growing model into bert model
        bert_model.load_state_dict(state)

        # compare the function of the two transformers
        bert_model.eval()
        growing_model.eval()

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        max_seq_length = 64

        raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

        assert "train" in raw_datasets, "Dataset does not contain train data."

        # remove headings
        def is_heading(text):
            text = text.strip()
            return text.startswith("=") and text.endswith("=")

        raw_datasets = raw_datasets.filter(lambda sample: not is_heading(sample["text"]))

        tokenized_datasets = prepare_mlm_dataset(
            tokenizer=tokenizer,
            dataset=raw_datasets,
            ignore_cache=False,
            max_seq_length=max_seq_length,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        data_loader = torch.utils.data.DataLoader(
            tokenized_datasets["train"], batch_size=1, shuffle=False, collate_fn=data_collator
        )

        for batch in data_loader:

            # ignore labels
            del batch["labels"]

            # === Test embeddings ===

            input_ids = batch["input_ids"]

            embeddings_a = growing_model.embeddings(
                input_ids=input_ids,
            )

            embeddings_b = bert_model.embeddings(
                input_ids=input_ids,
            )

            diff = embeddings_a - embeddings_b

            log.info(f"Max. embed difference: {diff.max()}")

            assert torch.all(diff < 1e-5)

            # === Test Transformer Model ===

            y_a = growing_model(**batch)
            y_b = bert_model(**batch)[0]

            diff = y_a - y_b

            log.info(f"Max. difference: {diff.max()}")

            assert torch.all(diff < 1e-5)

            break
