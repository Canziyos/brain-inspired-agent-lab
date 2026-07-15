from dataclasses import dataclass
from enum import StrEnum


class EventType(StrEnum):
    RESTED = "rested"
    ATE_FOOD = "ate_food"
    HIT_DANGER = "hit_danger"
    DISCOVERED_MYSTERY = "discovered_mystery"
    VISITED_EMPTY = "visited_empty"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    new_position: tuple[int, int]
    event: EventType
    energy_change: float
    health_change: float
    curiosity_change: float