from dataclasses import dataclass
from typing import Dict, List, Optional


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
    tune_step_size: bool
    selection_method: str
    grow_data_portion: float
    grow_tune_params: Dict


@dataclass
class Configuration:
    dataset: Dataset
    model: Model
    training: Training
    save_model: bool
    load_state: Optional[str] = None
