import logging
import logging.config

import torch

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"growing_transformer": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
)

logger = logging.getLogger("growing_transformer")


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from .configuration import GrowingConfig  # noqa: E402 import after setting device
from .model import (
    Growing,
    GrowingMLMTransformer,
    GrowingModule,
    GrowingTransformer,
)  # noqa: E402 import after setting device
from .trainer import BaseTrainer, GrowingTrainer, GrowthSchedule  # noqa: E402 import after setting device

__all__ = [
    "Growing",
    "GrowingModule",
    "GrowingConfig",
    "GrowingTransformer",
    "GrowingMLMTransformer",
    "BaseTrainer",
    "GrowingTrainer",
    "GrowthSchedule",
]
