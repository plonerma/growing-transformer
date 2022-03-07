import logging
import logging.config

from .base import Growing, GrowingModule
from .mlp import MLP
from .multiheadAttention import MultiheadAttention
from .scaledDotProductAttention import ScaledDotProductAttention
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
