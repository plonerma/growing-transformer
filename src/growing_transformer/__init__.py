import logging
import logging.config

from .base import Growing, GrowingModule
from .configuration import GrowingConfig
from .trainer import Trainer
from .transformer import GrowingTransformer

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


__all__ = [
    "Growing",
    "GrowingModule",
    "GrowingConfig",
    "GrowingTransformer",
    "Trainer",
]
