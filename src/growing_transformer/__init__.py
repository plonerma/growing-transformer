import logging
import logging.config

from .configuration import GrowingConfig
from .model import Growing, GrowingModule, GrowingTransformer
from .trainer import Trainer

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
