import logging
import random
from typing import Union

import datasets
import hydra
import torch
from config import Configuration
from datasets import DatasetDict
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling

import growing_transformer
from growing_transformer import GrowingConfig, GrowingTrainer, GrowthSchedule
from growing_transformer.data import (
    downsample_dataset,
    group_dataset,
    split_dataset,
    tokenize_dataset,
)
from growing_transformer.model import GrowingMLMTransformer

cs = ConfigStore.instance()
cs.store(name="base_config", node=Configuration)


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

        # with accelerator.main_process_first():
        tokenized_datasets = tokenize_dataset(
            tokenizer, raw_datasets, num_workers=cfg.preprocessing_num_workers, ignore_cache=cfg.ignore_cache
        )

        tokenized_datasets = group_dataset(
            tokenized_datasets,
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

    model: Union[GrowingMLMTransformer, BertForMaskedLM]

    if cfg.model.type == "growing":
        model = GrowingMLMTransformer(model_config)

    elif cfg.model.type == "huggingface":
        model = BertForMaskedLM(model_config)

    else:
        raise RuntimeError(f"Model variant {cfg.model.type} not implemented.")

    if cfg.load_state is not None:
        state_dict = torch.load(cfg.load_state, map_location=growing_transformer.device)
        model.load_state_dict(state_dict)

    tensorboard_writer = SummaryWriter(".")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.training.mlm_probability)

    trainer = GrowingTrainer(
        model,
        tune_direction=cfg.training.tune_direction,
        tune_step_size=cfg.training.tune_step_size,
        selection_method=cfg.training.selection_method,
        data_collator=data_collator,
    )

    trainer.train(
        train_data=train_data,
        grow_data_portion=cfg.training.grow_data_portion,
        schedule=schedule,
        test_data=test_data,
        tensorboard_writer=tensorboard_writer,
        batch_size=cfg.training.batch_size,
        gca_batches=cfg.training.gca_batches,
        max_lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
        use_onecycle=cfg.training.use_onecycle,
        warmup_pct=cfg.training.warmup_pct,
        grow_tune_params=cfg.training.grow_tune_params,
    )

    if cfg.save_model:
        torch.save(model.state_dict(), "trained_model.pt")

    tensorboard_writer.close()


if __name__ == "__main__":
    main()
