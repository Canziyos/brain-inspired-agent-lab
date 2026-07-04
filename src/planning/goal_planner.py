from dataclasses import dataclass
from enum import Enum

from src.core.agent import Agent
from src.core.world import FOOD, MYSTERY
from src.planning.frontier_planner import (
    Position,
    find_frontiers,
    known_traversable_cells,
    shortest_path,
)


class GoalKind(str, Enum):
    FOOD = "food"
    MYSTERY = "mystery"
    FRONTIER = "frontier"


@dataclass(frozen=True, slots=True)
class GoalPlan:
    kind: GoalKind
    target: Position
    path: tuple[Position, ...]

    @property
    def next_step(self) -> Position:
        if len(self.path) < 2:
            raise ValueError(
                "A usable plan must contain at least "
                "the current and next positions."
            )

        return self.path[1]


def known_object_targets(
    agent: Agent,
    cell_type: str,
) -> set[Position]:
    return {
        position
        for position, known_cell in agent.known_cells.items()
        if known_cell == cell_type
    }


def build_plan(
    kind: GoalKind,
    targets: set[Position],
    agent: Agent,
    traversable: set[Position],
    width: int,
    height: int,
) -> GoalPlan | None:
    if not targets:
        return None

    path = shortest_path(
        start=(agent.x, agent.y),
        goals=targets,
        traversable=traversable,
        width=width,
        height=height,
    )

    if path is None or len(path) < 2:
        return None

    return GoalPlan(
        kind=kind,
        target=path[-1],
        path=tuple(path),
    )


def select_goal_plan(
    agent: Agent,
    width: int,
    height: int,
) -> GoalPlan | None:
    traversable = known_traversable_cells(agent)

    food_targets = known_object_targets(
        agent,
        FOOD,
    )

    mystery_targets = known_object_targets(
        agent,
        MYSTERY,
    )

    frontier_targets = find_frontiers(
        agent,
        width,
        height,
    )

    # Food becomes urgent when energy is no longer high.
    if agent.energy < 85.0:
        priority_groups = (
            (GoalKind.FOOD, food_targets),
            (GoalKind.MYSTERY, mystery_targets),
            (GoalKind.FRONTIER, frontier_targets),
        )
    else:
        priority_groups = (
            (GoalKind.MYSTERY, mystery_targets),
            (GoalKind.FOOD, food_targets),
            (GoalKind.FRONTIER, frontier_targets),
        )

    for kind, targets in priority_groups:
        plan = build_plan(
            kind=kind,
            targets=targets,
            agent=agent,
            traversable=traversable,
            width=width,
            height=height,
        )

        if plan is not None:
            return plan

    return None


def has_reachable_goal(
    agent: Agent,
    width: int,
    height: int,
) -> bool:
    traversable = known_traversable_cells(agent)

    goal_groups = (
        known_object_targets(agent, FOOD),
        known_object_targets(agent, MYSTERY),
        find_frontiers(agent, width, height),
    )

    for targets in goal_groups:
        path = shortest_path(
            start=(agent.x, agent.y),
            goals=targets,
            traversable=traversable,
            width=width,
            height=height,
        )

        if path is not None:
            return True

    return False


def is_task_complete(
    agent: Agent,
    width: int,
    height: int,
) -> bool:
    return not has_reachable_goal(
        agent=agent,
        width=width,
        height=height,
    )