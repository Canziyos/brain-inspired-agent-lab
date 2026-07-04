from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import SimulationConfig
from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics import apply_action
from src.core.perception import Observation
from src.core.world import DANGER, EMPTY, FOOD, MYSTERY, World
from src.learning.rewards import calculate_reward
from src.simulation.setup import create_world
from src.planning.goal_planner import (
    is_task_complete,
)

BOUNDARY_INDEX = 4
CELL_TO_INDEX = {
    EMPTY: 0,
    FOOD: 1,
    DANGER: 2,
    MYSTERY: 3,
}

ACTION_REST = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4

ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_RIGHT: (1, 0),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
}

DELTA_TO_ACTION = {
    delta: action
    for action, delta in ACTION_DELTAS.items()
}


def action_to_discrete(
    position: tuple[int, int],
    action: Action,
) -> int:
    if action.kind == "rest":
        return ACTION_REST

    if action.kind != "move":
        raise ValueError(f"Unsupported action kind: {action.kind}")

    dx = action.target_x - position[0]
    dy = action.target_y - position[1]

    try:
        return DELTA_TO_ACTION[(dx, dy)]
    except KeyError as exc:
        raise ValueError(
            "Only cardinal one-cell moves can be mapped "
            "to the GridWorld action space."
        ) from exc


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

        self.config = config or SimulationConfig(
            verbose=False,
            show_animation=False,
            show_plots=False,
        )
        if (
            render_mode is not None
            and render_mode not in self.metadata["render_modes"]
        ):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self.cell_size = cell_size

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array(
                        [
                            self.config.world_width - 1,
                            self.config.world_height - 1,
                        ],
                        dtype=np.int32,
                    ),
                    dtype=np.int32,
                ),
                "internal": spaces.Box(
                    low=0.0,
                    high=100.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "neighbors": spaces.MultiDiscrete([5, 5, 5, 5]),
            }
        )

        self.world: World | None = None
        self.agent: Agent | None = None
        self.step_count = 0

        self._pygame: Any | None = None
        self._window: Any | None = None
        self._clock: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        config = self.config
        if seed is not None:
            config = replace(
                config,
                random_seed=seed,
            )

        self.world = create_world(config)
        self.agent = Agent(x=0, y=0)
        self.step_count = 0
        self._update_agent_belief()

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

        physical_action = self._to_physical_action(action)
        outcome = apply_action(
            self.agent,
            self.world,
            physical_action,
        )
        reward = calculate_reward(outcome)

        self.step_count += 1

        self._update_agent_belief()

        dead = not self.agent.is_alive()

        task_complete = (
            not dead
            and is_task_complete(
                agent=self.agent,
                width=self.world.width,
                height=self.world.height,
            )
        )

        terminated = dead or task_complete

        truncated = (
            self.step_count >= self.config.max_steps
            and not terminated
        )

        observation = self._get_observation()
        info = self._get_info()
        info["event"] = outcome.event.name
        if dead:
            termination_reason = "dead"
        elif task_complete:
            termination_reason = "goals_complete"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = None

        info["task_complete"] = task_complete
        info["termination_reason"] = termination_reason

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        pygame = self._load_pygame()
        width_px = self.world.width * self.cell_size
        height_px = self.world.height * self.cell_size

        if self.render_mode == "human":
            if self._window is None:
                self._window = pygame.display.set_mode(
                    (width_px, height_px)
                )
                pygame.display.set_caption("Baby Vice GridWorld")
            surface = self._window
        else:
            surface = pygame.Surface((width_px, height_px))

        self._draw(surface)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self._clock is None:
                self._clock = pygame.time.Clock()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            pygame.surfarray.array3d(surface),
            axes=(1, 0, 2),
        )

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._window = None
        self._clock = None

    def _update_agent_belief(self) -> None:
        self._require_reset()

        assert self.agent is not None

        self.agent.observe(
            self._get_symbolic_observations()
        )

    def _to_physical_action(
        self,
        action: int,
    ) -> Action:
        assert self.agent is not None
        assert self.world is not None

        if action == ACTION_REST:
            return Action(
                kind="rest",
                target_x=self.agent.x,
                target_y=self.agent.y,
            )

        dx, dy = ACTION_DELTAS[action]
        target_x = self.agent.x + dx
        target_y = self.agent.y + dy

        if not self.world.is_inside(target_x, target_y):
            return Action(
                kind="blocked",
                target_x=self.agent.x,
                target_y=self.agent.y,
            )

        return Action(
            kind="move",
            target_x=target_x,
            target_y=target_y,
        )

    def _get_observation(self) -> dict[str, np.ndarray]:
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        neighbors = []
        for dx, dy in [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
        ]:
            cell = self.world.get_cell(
                self.agent.x + dx,
                self.agent.y + dy,
            )
            if cell is None:
                neighbors.append(BOUNDARY_INDEX)
            else:
                neighbors.append(CELL_TO_INDEX[cell])

        return {
            "position": np.array(
                [self.agent.x, self.agent.y],
                dtype=np.int32,
            ),
            "internal": np.array(
                [
                    self.agent.energy,
                    self.agent.health,
                    self.agent.curiosity,
                ],
                dtype=np.float32,
            ),
            "neighbors": np.array(
                neighbors,
                dtype=np.int64,
            ),
        }

    def _get_info(self) -> dict[str, Any]:
        self._require_reset()
        assert self.world is not None
        assert self.agent is not None

        return {
            "step": self.step_count,
            "position": (self.agent.x, self.agent.y),
            "observations": tuple(self._get_symbolic_observations()),
            "grid": tuple(
                tuple(row)
                for row in self.world.grid
            ),
        }

    def _get_symbolic_observations(self) -> list[Observation]:
        assert self.world is not None
        assert self.agent is not None

        return [
            Observation(x=x, y=y, cell=cell)
            for x, y, cell in self.world.neighbors(
                self.agent.x,
                self.agent.y,
            )
        ]

    def _draw(self, surface) -> None:
        assert self.world is not None
        assert self.agent is not None

        pygame = self._load_pygame()
        colors = {
            EMPTY: (236, 236, 236),
            FOOD: (91, 174, 87),
            DANGER: (191, 72, 72),
            MYSTERY: (116, 105, 182),
        }

        surface.fill((32, 34, 37))

        for y, row in enumerate(self.world.grid):
            for x, cell in enumerate(row):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(
                    surface,
                    colors.get(cell, colors[EMPTY]),
                    rect,
                )
                pygame.draw.rect(
                    surface,
                    (45, 48, 51),
                    rect,
                    1,
                )

        center = (
            self.agent.x * self.cell_size
            + self.cell_size // 2,
            self.agent.y * self.cell_size
            + self.cell_size // 2,
        )
        pygame.draw.circle(
            surface,
            (35, 87, 164),
            center,
            self.cell_size // 3,
        )

    def _load_pygame(self):
        if self._pygame is None:
            import pygame

            pygame.init()
            self._pygame = pygame
        return self._pygame

    def _require_reset(self) -> None:
        if self.world is None or self.agent is None:
            raise RuntimeError("Call reset() before step() or render().")
