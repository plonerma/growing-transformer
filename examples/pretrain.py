import logging
import random
from pathlib import Path
from typing import Any, Dict, Union

import datasets
import hydra
import torch
from config import Configuration
from datasets import DatasetDict
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling

import growing_transformer
from growing_transformer import GrowingConfig, GrowingTrainer, GrowthSchedule
from growing_transformer.data import (
    downsample_dataset,
    prepare_mlm_dataset,
    split_dataset,
)
from growing_transformer.model import GrowingMLMTransformer

cs = ConfigStore.instance()
cs.store(name="base_config", node=Configuration)


use_truncated_normal = True


def truncated_normal_(tensor, mean=0, std=1):
    """Source: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16"""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class HuggingfaceMLMTransformer(BertForMaskedLM):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@hydra.main(config_path="config", config_name="config")
def main(cfg: Configuration):
    log = logging.getLogger("growing_transformer")
    # add_file_handler(log, Path("training.log"))

    growing_transformer.device = torch.device(cfg.training.device)

    # mark as used
    _ = torch.tensor(0).to(growing_transformer.device)

    model_config = GrowingConfig(**cfg.model.config)
    schedule = GrowthSchedule(cfg.training.schedule["steps"])

    log.info(model_config)
    log.info(schedule)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.tokenizer)

    loaded_datasets = list()

    assert len(cfg.datasets) > 0, "No datasets selected."

    if cfg.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            log.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if cfg.max_seq_length > tokenizer.model_max_length:
            log.warning(
                f"The max_seq_length passed ({cfg.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(cfg.max_seq_length, tokenizer.model_max_length)

    for dataset_cfg in cfg.datasets.values():
        log.info(f"Loading dataset {dataset_cfg.name} / {dataset_cfg.version}")
        raw_datasets = datasets.load_dataset(dataset_cfg.name, dataset_cfg.version)

        assert isinstance(raw_datasets, DatasetDict)
        assert "train" in raw_datasets, "Dataset does not contain train data."

        if dataset_cfg.name == "wikitext":
            # remove headings
            def is_heading(text):
                text = text.strip()
                return text.startswith("=") and text.endswith("=")

            raw_datasets = raw_datasets.filter(lambda sample: not is_heading(sample["text"]))

        tokenized_datasets = prepare_mlm_dataset(
            tokenizer=tokenizer,
            dataset=raw_datasets,
            num_workers=cfg.preprocessing_num_workers,
            ignore_cache=cfg.ignore_cache,
            max_seq_length=max_seq_length,
        )

        train = tokenized_datasets["train"]

        if dataset_cfg.downsample is not None and dataset_cfg.downsample < 1:
            train = downsample_dataset(train, dataset_cfg.downsample)

        if "test" in tokenized_datasets:
            test = tokenized_datasets["test"]

        else:
            log.info(f"Dataset has not test-set using {dataset_cfg.test_portion} of train data as test set.")
            assert dataset_cfg.test_portion is not None
            # set seed for test split
            random.seed(dataset_cfg.test_split_seed)
            test, train = split_dataset(train, dataset_cfg.test_portion)

        loaded_datasets.append((train, test))

    train_data: torch.utils.data.Dataset
    test_data: torch.utils.data.Dataset

    if len(loaded_datasets) > 1:
        train_datasets, test_datasets = zip(*loaded_datasets)
        train_data = torch.utils.data.ConcatDataset(train_datasets)
        test_data = torch.utils.data.ConcatDataset(test_datasets)
    else:
        train_data, test_data = loaded_datasets[0]

    model: torch.nn.Module

    if cfg.model.type == "growing":
        model = GrowingMLMTransformer(model_config)

    elif cfg.model.type == "huggingface":
        if use_truncated_normal:
            model = HuggingfaceMLMTransformer(model_config)
        else:
            model = BertForMaskedLM(model_config)

    else:
        raise RuntimeError(f"Model variant {cfg.model.type} not implemented.")

    if cfg.load_state is not None:
        model_path = Path(get_original_cwd()) / cfg.load_state
        state_dict = torch.load(model_path, map_location=growing_transformer.device)
        model.load_state_dict(state_dict)

    tensorboard_writer = SummaryWriter(".")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.training.mlm_probability)

    hparams_init: Dict[str, Any] = dict(
        batch_size=cfg.training.batch_size,
        tune_direction=cfg.training.tune_direction,
        tune_step_size=cfg.training.tune_step_size,
        selection_method=cfg.training.selection_method,
    )

    hparams_train: Dict[str, Any] = dict(
        grow_data_portion=cfg.training.grow_data_portion,
        gca_batches=cfg.training.gca_batches,
        eps=cfg.training.eps,
        max_lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    def custom_eval(self, global_step=None):
        num_examples = 8

        with torch.no_grad():
            batch_loader = self.get_batch_loader(test_data, batch_size=num_examples, num_workers=0, shuffle=False)

            for batch in batch_loader:
                batch = self.prepare_batch(batch)

                outputs = self.model(**batch)
                prediction_scores = outputs[1]
                predictions = prediction_scores.argmax(-1)

                labels = batch["labels"]
                mlm_mask = labels >= 0

                masked_sentence = tokenizer.batch_decode(batch["input_ids"])
                correct_sentence = tokenizer.batch_decode(torch.where(mlm_mask, labels, batch["input_ids"]))
                predicted_sentence = tokenizer.batch_decode(torch.where(mlm_mask, predictions, batch["input_ids"]))

                for i in range(num_examples):
                    tensorboard_writer.add_text(f"example {i}, masked sentence", masked_sentence[i], global_step)
                    tensorboard_writer.add_text(f"example {i}, correct sentence", correct_sentence[i], global_step)
                    tensorboard_writer.add_text(f"example {i}, predicted sentence", predicted_sentence[i], global_step)

                break

    trainer = GrowingTrainer(model, data_collator=data_collator, custom_eval=custom_eval, **hparams_init)

    results = trainer.train(
        train_data=train_data,
        schedule=schedule,
        test_data=test_data,
        tensorboard_writer=tensorboard_writer,
        betas=cfg.training.betas,
        grow_tune_params=cfg.training.grow_tune_params,
        quit_on_step=cfg.total_steps,
        **hparams_train,
    )

    tensorboard_writer.add_hparams({**hparams_init, **hparams_train}, results)

    if cfg.save_model:
        torch.save(model.state_dict(), "trained_model.pt")

    tensorboard_writer.close()


if __name__ == "__main__":
    main()
