from .base import Growing, GrowingModule
from .mlp import MLP
from .scaledDotProductAttention import ScaledDotProductAttention
from .transformer import GrowingTransformer
from .multiheadAttention import MultiheadAttention
from .trainer import Trainer

import logging
import logging.config


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
