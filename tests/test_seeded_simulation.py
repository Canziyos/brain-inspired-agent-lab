from src.config import SimulationConfig
from src.simulation.runner import run_simulation


def make_config() -> SimulationConfig:
    return SimulationConfig(
        world_width=16,
        world_height=12,
        food_count=5,
        danger_count=8,
        mystery_count=10,
        max_steps=20,
        random_seed=7,
        torch_seed=7,
        verbose=False,
        show_animation=False,
        show_plots=False,
    )


def simulation_signature(history) -> list[tuple]:
    return [
        (
            item.position,
            item.reward,
            item.event,
            item.visited_count,
            item.known_cell_count,
            item.rule_action,
            item.action_reason,
        )
        for item in history
    ]


def test_seeded_simulation_is_reproducible() -> None:
    history_a = run_simulation(make_config())
    history_b = run_simulation(make_config())

    assert len(history_a) == 20
    assert len(history_b) == 20

    assert simulation_signature(
        history_a
    ) == simulation_signature(
        history_b
    )


def test_simulation_metrics_remain_valid() -> None:
    config = make_config()
    history = run_simulation(config)

    assert all(
        0 <= item.position[0] < config.world_width
        and 0 <= item.position[1] < config.world_height
        for item in history
    )

    visited_counts = [
        item.visited_count
        for item in history
    ]

    known_counts = [
        item.known_cell_count
        for item in history
    ]

    assert visited_counts == sorted(visited_counts)
    assert known_counts == sorted(known_counts)

    assert all(
        0.0 <= item.energy <= 100.0
        and 0.0 <= item.health <= 100.0
        and 0.0 <= item.curiosity <= 100.0
        for item in history
    )
