from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Dataset:
    name: str
    version: str
    downsample: Optional[float] = None
    test_portion: Optional[float] = None
    test_split_seed: Optional[int] = 0
    num_workers: Optional[int] = None


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
    mlm_probability: float = 0.15


@dataclass
class Configuration:
    datasets: Dict[str, Dataset]
    model: Model
    training: Training
    save_model: bool
    load_state: Optional[str] = None
    ignore_cache: bool = False
    preprocessing_num_workers: Optional[int] = 1
    max_seq_length: Optional[int] = None
