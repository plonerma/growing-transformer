from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Dataset:
    name: str
    version: str
    downsample: float


@dataclass
class Model:
    type: str
    config: Dict
    tokenizer: str


@dataclass
class Training:
    schedule: Dict
    learning_rate: float
    batch_size: int
    gca_batches: int
    device: str
    betas: List[float]
    eps: float
    weight_decay: float
    use_onecycle: bool
    growth_phases: int
    tune_direction: bool
    tune_new_parts: bool
    selection_method: str


@dataclass
class Configuration:
    dataset: Dataset
    model: Model
    training: Training
