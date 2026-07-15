from dataclasses import dataclass

from src.core.agent import Agent
from src.core.world import CellType, World


@dataclass(frozen=True, slots=True)
class TerminationStatus:
    terminated: bool
    truncated: bool
    task_complete: bool
    reason: str | None


def is_world_task_complete(world: World) -> bool:
    return all(
        cell not in {
            CellType.FOOD,
            CellType.MYSTERY,
        }
        for row in world.grid
        for cell in row
    )


def determine_termination(
    agent: Agent,
    world: World,
    step_count: int,
    max_steps: int,
) -> TerminationStatus:
    dead = not agent.is_alive()
    task_complete = (
        not dead
        and is_world_task_complete(world)
    )

    terminated = dead or task_complete

    truncated = (
        step_count >= max_steps
        and not terminated
    )

    if dead:
        reason = "dead"
    elif task_complete:
        reason = "goals_complete"
    elif truncated:
        reason = "time_limit"
    else:
        reason = None

    return TerminationStatus(
        terminated=terminated,
        truncated=truncated,
        task_complete=task_complete,
        reason=reason,
    )