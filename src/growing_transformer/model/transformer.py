from ..configuration import GrowingConfig
from .base import Growing


class GrowingTransformer(Growing):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)
        raise NotImplementedError
