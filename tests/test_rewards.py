from src.core.dynamics_types import ActionOutcome, EventType
from src.learning.rewards import calculate_reward


def test_calculate_reward_weights_health_more_than_energy() -> None:
    reward = calculate_reward(
        ActionOutcome(
            new_position=(0, 0),
            event=EventType.HIT_DANGER,
            energy_change=-7.0,
            health_change=-30.0,
            curiosity_change=0.0,
        )
    )

    assert reward == -97.0


def test_calculate_reward_penalizes_empty_cell_visit() -> None:
    reward = calculate_reward(
        ActionOutcome(
            new_position=(0, 0),
            event=EventType.VISITED_EMPTY,
            energy_change=-2.0,
            health_change=0.0,
            curiosity_change=0.0,
        )
    )

    assert reward == -3.0
