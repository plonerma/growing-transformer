from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence

from .. import GrowingModule


@dataclass
class GrowthPhase:
    epochs: int  # epochs to train after growing
    params: dict = field(default_factory=dict)
    config_update: dict = field(default_factory=dict)
    index: Optional[int] = None

    def grow_params(self, m: GrowingModule):
        for cls in m.__class__.mro():
            try:
                return self.params[cls]["grow"]
            except KeyError:
                continue

    def num_kept_parts(self, m: GrowingModule):
        for cls in m.__class__.mro():
            try:
                return self.params[cls]["num_kept_parts"]
            except KeyError:
                continue

    @property
    def is_initial(self) -> bool:
        return self.index is not None and self.index == 0


class StepType(Enum):
    train = 0
    grow = 1


class GrowthSchedule:
    def __init__(self, steps: Sequence):
        self.steps = list()

        for step_spec in steps:
            if isinstance(step_spec, Mapping):
                assert len(step_spec) == 1, "Step should have exactly one key."
                ((step_type, step_params),) = step_spec.items()
            else:
                assert len(step_spec) == 2, "Step should have exactly have type and params."
                step_type, step_params = step_spec

            step_type = StepType[step_type.lower()]

            self.steps.append((step_type, step_params))

    def __iter__(self):
        return iter(self.steps)

    def __str__(self):
        entries = "".join([f"  {step},\n" for step in self])
        return f"{self.__class__.__name__} [\n{entries}]"

    def __len__(self):
        return len(self.steps)
