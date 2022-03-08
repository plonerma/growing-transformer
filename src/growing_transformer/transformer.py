from typing import Any, Mapping

import torch


class GrowingTransformer(torch.nn.Module):
    def __init__(self, *, config: Mapping[str, Any]):
        raise NotImplementedError
