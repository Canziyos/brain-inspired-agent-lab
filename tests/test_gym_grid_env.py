from dataclasses import replace

import numpy as np

from src.configs import SimulationConfig
from src.core.actions import Action
from src.core.world import CellType
from src.envs import BabyViceGridEnv
from src.envs.grid_world_actions import (
    ACTION_DOWN,
    ACTION_REST,
    ACTION_RIGHT,
    action_to_discrete,
)


def make_env(render_mode=None) -> BabyViceGridEnv:
    base_config = SimulationConfig()
    return BabyViceGridEnv(
        config=replace(
            base_config,
            runtime=replace(
                base_config.runtime,
                max_steps=3,
            ),
        ),
        render_mode=render_mode,
        cell_size=8,
    )


def test_reset_is_seed_deterministic() -> None:
    env = make_env()

    observation_a, info_a = env.reset(seed=42)
    observation_b, info_b = env.reset(seed=42)

    assert env.observation_space.contains(observation_a)
    assert env.observation_space.contains(observation_b)
    assert np.array_equal(
        observation_a["position"],
        observation_b["position"],
    )
    assert info_a["grid"] == info_b["grid"]
    assert info_a["grid"][0][0] is CellType.EMPTY
    assert info_a["observations"] == info_b["observations"]


def test_step_uses_gymnasium_contract() -> None:
    env = make_env()
    env.reset(seed=7)

    observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(observation)
    assert reward == 5.0
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert info["event"] == "RESTED"
    assert info["observations"]


def test_truncates_at_configured_max_steps() -> None:
    env = make_env()
    env.reset(seed=7)

    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(0)

    assert truncated is True


def test_rgb_array_render_is_headless() -> None:
    env = make_env(render_mode="rgb_array")
    env.reset(seed=7)

    frame = env.render()

    assert frame.shape == (
        env.config.world.height * env.cell_size,
        env.config.world.width * env.cell_size,
        3,
    )
    assert frame.dtype == np.uint8

    env.close()


def test_physical_action_maps_to_discrete_actuator() -> None:
    assert action_to_discrete(
        (0, 0),
        Action.REST,
    ) == ACTION_REST
    assert action_to_discrete(
        (0, 0),
        Action.MOVE_EAST,
    ) == ACTION_RIGHT
    assert action_to_discrete(
        (0, 0),
        Action.MOVE_SOUTH,
    ) == ACTION_DOWN
