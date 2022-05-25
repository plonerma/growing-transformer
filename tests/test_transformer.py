import datasets
import pytest
import torch
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling

from growing_transformer import GrowingConfig, GrowingMLMTransformer
from growing_transformer.data import prepare_mlm_dataset


class TestGrowingMLMTransformer:
    @pytest.mark.slow
    def test_transformer_function(self):
        # initialize growing transformer
        config = GrowingConfig()
        growing_model = GrowingMLMTransformer(config)

        # get state from that model
        state = growing_model.state_dict()

        # initialize bert transformer
        bert_model = BertForMaskedLM(config)

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

        for i, batch in enumerate(data_loader):
            if i == 10:
                break

            y_a = growing_model(**batch)
            y_b = bert_model(**batch)

            diff = y_a["logits"] - y_b["logits"]

            log.info(f"Max. difference: {diff.max()}")

            assert torch.all(diff < 1e-5)
