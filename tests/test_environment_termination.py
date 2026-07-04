from src.config import SimulationConfig
from src.envs.grid_world_env import (
    ACTION_REST,
    BabyViceGridEnv,
)


def test_environment_terminates_on_goal_completion() -> None:
    config = SimulationConfig(
        world_width=2,
        world_height=1,
        food_count=0,
        danger_count=0,
        mystery_count=0,
        max_steps=10,
        verbose=False,
        show_animation=False,
        show_plots=False,
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
