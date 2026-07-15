from dataclasses import replace

from src.configs import SimulationConfig
from src.envs.grid_world_actions import (
    ACTION_REST,
)
from src.envs.grid_world_env import BabyViceGridEnv


def test_environment_terminates_on_goal_completion() -> None:
    base_config = SimulationConfig()
    config = replace(
        base_config,
        world=replace(
            base_config.world,
            width=2,
            height=1,
            food_count=0,
            danger_count=0,
            mystery_count=0,
        ),
        runtime=replace(
            base_config.runtime,
            max_steps=10,
        ),
    )

    env = BabyViceGridEnv(config=config)

    try:
        env.reset(seed=1)

        (
            _observation,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(ACTION_REST)

        assert terminated
        assert not truncated
        assert info["task_complete"]
        assert (
            info["termination_reason"]
            == "goals_complete"
        )

    finally:
        env.close()
