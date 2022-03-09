from .base import Growing
from .configuration import GrowingConfig


class GrowingTransformer(Growing):
    def __init__(self, config: GrowingConfig):
        super().__init__(config=config)
        raise NotImplementedError
