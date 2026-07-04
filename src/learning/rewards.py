from src.core.agent import Experience
from src.core.dynamics_types import ActionOutcome, EventType


def calculate_reward(
    outcome: Experience | ActionOutcome,
) -> float:
    reward = (
        outcome.energy_change
        + 3.0 * outcome.health_change
    )

    if outcome.event == EventType.DISCOVERED_MYSTERY:
        reward += 8.0

    elif outcome.event == EventType.VISITED_EMPTY:
        reward -= 1.0
    elif outcome.event == EventType.BLOCKED:
        reward -= 3.0

    return reward
