from typing import Final

from src.core.dynamics_types import ActionOutcome, EventType


HEALTH_REWARD_WEIGHT: Final[float] = 3.0

EVENT_REWARD_ADJUSTMENTS: Final[dict[EventType, float]] = {
    EventType.DISCOVERED_MYSTERY: 8.0,
    EventType.VISITED_EMPTY: -1.0,
    EventType.BLOCKED: -3.0,
}


def event_reward_adjustment(event: EventType) -> float:
    return EVENT_REWARD_ADJUSTMENTS.get(event, 0.0)


def calculate_reward(outcome: ActionOutcome) -> float:
    return (
        outcome.energy_change
        + HEALTH_REWARD_WEIGHT * outcome.health_change
        + event_reward_adjustment(outcome.event)
    )