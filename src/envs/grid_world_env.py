from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.configs import SimulationConfig
from src.core.agent import Agent
from src.core.dynamics import apply_action
from src.core.world import World
from src.envs.world_factory import create_world
from src.envs.grid_world_actions import (
    ACTION_COUNT,
    action_to_discrete,
    discrete_to_action,
)
from src.envs.grid_world_observation import (
    build_base_info,
    build_observation,
    build_step_info,
    create_observation_space,
)
from src.envs.grid_world_render import GridWorldRenderer
from src.envs.grid_world_termination import (
    determine_termination,
)
from src.learning.rewards import calculate_reward


class BabyViceGridEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 8,
    }

    def __init__(
        self,
        config: SimulationConfig | None = None,
        render_mode: str | None = None,
        cell_size: int = 64,
    ) -> None:
        super().__init__()

        self.config = config or SimulationConfig()

        if (
            render_mode is not None
            and render_mode not in self.metadata["render_modes"]
        ):
            raise ValueError(
                f"Unsupported render mode: {render_mode}"
            )

        self.render_mode = render_mode
        self.cell_size = cell_size

        self.action_space = spaces.Discrete(
            ACTION_COUNT
        )

        self.observation_space = create_observation_space(
            width=self.config.world.width,
            height=self.config.world.height,
        )

        self.world: World | None = None
        self.agent: Agent | None = None
        self.step_count = 0

        self._renderer = GridWorldRenderer(
            render_mode=render_mode,
            cell_size=cell_size,
            render_fps=self.metadata["render_fps"],
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options

        super().reset(seed=seed)

        config = self.config

        if seed is not None:
            config = replace(
                config,
                runtime=replace(
                    config.runtime,
                    random_seed=seed,
                ),
            )

        self.world = create_world(
            config=config.world,
            random_seed=config.runtime.random_seed,
        )
        self.agent = Agent(x=0, y=0)
        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        physical_action = discrete_to_action(action)

        outcome = apply_action(
            self.agent,
            self.world,
            physical_action,
        )

        reward = calculate_reward(outcome)

        self.step_count += 1

        termination = determine_termination(
            agent=self.agent,
            world=self.world,
            step_count=self.step_count,
            max_steps=self.config.runtime.max_steps,
        )

        observation = self._get_observation()

        info = build_step_info(
            agent=self.agent,
            world=self.world,
            step_count=self.step_count,
            outcome=outcome,
            termination=termination,
        )

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            termination.terminated,
            termination.truncated,
            info,
        )

    def render(self):
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        return self._renderer.render(
            world=self.world,
            agent=self.agent,
        )

    def close(self) -> None:
        self._renderer.close()

    def _get_observation(
        self,
    ) -> dict[str, np.ndarray]:
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        return build_observation(
            agent=self.agent,
            world=self.world,
        )

    def _get_info(self) -> dict[str, Any]:
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        return build_base_info(
            agent=self.agent,
            world=self.world,
            step_count=self.step_count,
        )

    def _require_reset(self) -> None:
        if self.world is None or self.agent is None:
            raise RuntimeError(
                "Call reset() before step() or render()."
            )