from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    RESTED = auto()
    ATE_FOOD = auto()
    HIT_DANGER = auto()
    DISCOVERED_MYSTERY = auto()
    VISITED_EMPTY = auto()
    NO_OP = auto()
    BLOCKED = auto()


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    position: tuple[int, int]
    event: EventType
    energy_change: float
    health_change: float
    curiosity_change: float
