from dataclasses import dataclass, field
from typing import List, Optional

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


class GrowthSchedule:
    def __init__(self, intial_epochs: int = 1):
        self.initial_epochs = intial_epochs
        self.growth_phases: List[GrowthPhase] = list()

    def __iter__(self):
        yield GrowthPhase(self.intial_epochs, 0)
        yield from self.growth_phases

    def add_phase(self, *args, **kw):
        index = len(self.growth_phases) + 1
        self.growth_phases.append(GrowthPhase(*args, index=index, **kw))

    def __len__(self):
        return len(self.growth_phases)

    def total_epochs(self):
        return sum([phase.epochs for phase in self])
