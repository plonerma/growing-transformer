from dataclasses import dataclass, field
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
    tokenizer: str
    config: Dict = field(default_factory=dict)
    use_truncated_normal: bool = True


@dataclass
class Training:
    schedule: Dict
    learning_rate: float
    batch_size: int
    gca_batches: int
    betas: List[float]
    eps: float
    weight_decay: float
    tune_direction: bool
    tune_step_size: bool
    selection_method: str
    grow_data_portion: float
    grow_tune_params: Dict
    mlm_probability: float = 0.15
    num_workers: int = 4


@dataclass
class Configuration:
    device: str
    datasets: Dict[str, Dataset]
    model: Model
    training: Training
    save_model: bool
    load_state: Optional[str] = None
    ignore_cache: bool = False
    preprocessing_num_workers: Optional[int] = 1
    max_seq_length: Optional[int] = None
    total_steps: Optional[int] = None
    checkpoint_every: Optional[int] = None
