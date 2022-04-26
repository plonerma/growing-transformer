import logging
import logging.config

import torch

logger = logging.getLogger("growing_transformer")


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from .configuration import GrowingConfig  # noqa: E402 import after setting device
from .model import (  # noqa: E402 import after setting device
    Growing,
    GrowingMLMTransformer,
    GrowingModule,
    GrowingTransformer,
)
from .trainer import (  # noqa: E402 import after setting device
    BaseTrainer,
    GrowingTrainer,
    GrowthSchedule,
)

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
