from typing import Any

import numpy as np
from gymnasium import spaces

from src.core.agent import Agent
from src.core.dynamics_types import ActionOutcome
from src.core.perception import Observation
from src.core.world import CellType, World
from src.envs.grid_world_actions import MOVE_DELTAS
from src.envs.grid_world_termination import TerminationStatus


BOUNDARY_INDEX = 4

CELL_TO_INDEX = {
    CellType.EMPTY: 0,
    CellType.FOOD: 1,
    CellType.DANGER: 2,
    CellType.MYSTERY: 3,
}


def create_observation_space(
    width: int,
    height: int,
) -> spaces.Dict:
    return spaces.Dict(
        {
            "position": spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array(
                    [
                        width - 1,
                        height - 1,
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
            "neighbors": spaces.MultiDiscrete(
                [5] * len(MOVE_DELTAS)
            ),
        }
    )


def build_observation(
    agent: Agent,
    world: World,
) -> dict[str, np.ndarray]:
    neighbors: list[int] = []

    for dx, dy in MOVE_DELTAS:
        cell = world.get_cell(
            agent.x + dx,
            agent.y + dy,
        )

        if cell is None:
            neighbors.append(BOUNDARY_INDEX)
        else:
            neighbors.append(CELL_TO_INDEX[cell])

    return {
        "position": np.array(
            [agent.x, agent.y],
            dtype=np.int32,
        ),
        "internal": np.array(
            [
                agent.energy,
                agent.health,
                agent.curiosity,
            ],
            dtype=np.float32,
        ),
        "neighbors": np.array(
            neighbors,
            dtype=np.int64,
        ),
    }


def symbolic_observations(
    agent: Agent,
    world: World,
) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            x=x,
            y=y,
            cell=cell,
        )
        for x, y, cell in world.neighbors(
            agent.x,
            agent.y,
        )
    )


def build_base_info(
    agent: Agent,
    world: World,
    step_count: int,
) -> dict[str, Any]:
    return {
        "step": step_count,
        "position": agent.position,
        "observations": symbolic_observations(
            agent=agent,
            world=world,
        ),
        "grid": tuple(
            tuple(row)
            for row in world.grid
        ),
    }


def build_step_info(
    agent: Agent,
    world: World,
    step_count: int,
    outcome: ActionOutcome,
    termination: TerminationStatus,
) -> dict[str, Any]:
    info = build_base_info(
        agent=agent,
        world=world,
        step_count=step_count,
    )

    info.update(
        {
            "event": outcome.event.name,
            "energy_change": outcome.energy_change,
            "health_change": outcome.health_change,
            "curiosity_change": outcome.curiosity_change,
            "task_complete": termination.task_complete,
            "termination_reason": termination.reason,
        }
    )

    return info