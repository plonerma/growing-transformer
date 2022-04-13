import logging
from pathlib import Path

import datasets
import hydra
import torch
from config import Configuration
from datasets import DatasetDict
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

import growing_transformer
from growing_transformer import GrowingConfig, GrowingTrainer, GrowthSchedule
from growing_transformer.data import MLMSegmenetDataset
from growing_transformer.trainer.util import add_file_handler

cs = ConfigStore.instance()
cs.store(name="base_config", node=Configuration)


@hydra.main(config_path="config", config_name="config")
def main(cfg: Configuration):
    log = logging.getLogger("growing_transformer")
    add_file_handler(log, Path("training.log"))

    model_config = GrowingConfig(**cfg.model.config)
    schedule = GrowthSchedule(cfg.training.schedule["steps"])

    log.info(model_config)
    log.info(schedule)

    from growing_transformer import GrowingMLMTransformer

    model = GrowingMLMTransformer(model_config)
    for n, m in model.named_modules():
        print(n)

    quit(0)

    growing_transformer.device = torch.device(cfg.training.device)

    corpus = datasets.load_dataset(cfg.dataset.name, cfg.dataset.version)

    assert isinstance(corpus, DatasetDict)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.tokenizer)

    train_data = MLMSegmenetDataset(corpus["train"], tokenizer)  # .downsampled(0.1)
    if cfg.dataset.downsample < 1 - 1e-5:
        train_data = train_data.downsampled(cfg.dataset.downsample)

    test_data = MLMSegmenetDataset(corpus["test"], tokenizer)

    if cfg.model.type == "growing":
        from growing_transformer import GrowingMLMTransformer

        model = GrowingMLMTransformer(model_config)

    elif cfg.model.type == "huggingface":
        from growing_transformer.model import HuggingfaceMLMTransformer

        model = HuggingfaceMLMTransformer(model_config)

    else:
        raise RuntimeError(f"Model variant {cfg.model.variant} not implemented.")

    tensorboard_writer = SummaryWriter(".")

    trainer = GrowingTrainer(
        model,
        tune_direction=cfg.training.tune_direction,
        tune_new_parts=cfg.training.tune_new_parts,
        selection_method=cfg.training.selection_method,
    )

    trainer.train(
        train_data,
        grow_data=train_data,
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
    )

    torch.save(model.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    main()
