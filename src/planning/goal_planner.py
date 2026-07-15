from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

from src.core.agent import Agent
from src.core.motivation import (
    food_motivation,
    mystery_motivation,
)
from src.core.world import CellType
from src.planning.frontier_planner import (
    Position,
    find_frontiers,
    known_traversable_cells,
    shortest_path,
)


FRONTIER_MOTIVATION_SCORE = 14.0


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


@dataclass(frozen=True, slots=True)
class GoalCandidate:
    kind: GoalKind
    targets: frozenset[Position]
    priority: float


def known_object_targets(
    agent: Agent,
    cell_type: CellType,
) -> frozenset[Position]:
    return frozenset(
        position
        for position, known_cell in agent.known_cells.items()
        if known_cell is cell_type
    )


def build_plan(
    kind: GoalKind,
    targets: Collection[Position],
    agent: Agent,
    traversable: set[Position],
    width: int,
    height: int,
) -> GoalPlan | None:
    target_set = set(targets)

    if not target_set:
        return None

    path = shortest_path(
        start=agent.position,
        goals=target_set,
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


def goal_candidates(
    agent: Agent,
    width: int,
    height: int,
) -> tuple[GoalCandidate, ...]:
    state = agent.snapshot()

    return (
        GoalCandidate(
            kind=GoalKind.FOOD,
            targets=known_object_targets(
                agent,
                CellType.FOOD,
            ),
            priority=food_motivation(state),
        ),
        GoalCandidate(
            kind=GoalKind.MYSTERY,
            targets=known_object_targets(
                agent,
                CellType.MYSTERY,
            ),
            priority=mystery_motivation(state),
        ),
        GoalCandidate(
            kind=GoalKind.FRONTIER,
            targets=frozenset(
                find_frontiers(
                    agent,
                    width,
                    height,
                )
            ),
            priority=FRONTIER_MOTIVATION_SCORE,
        ),
    )


def ranked_goal_candidates(
    agent: Agent,
    width: int,
    height: int,
) -> tuple[GoalCandidate, ...]:
    return tuple(
        sorted(
            goal_candidates(
                agent,
                width,
                height,
            ),
            key=lambda candidate: candidate.priority,
            reverse=True,
        )
    )


def select_goal_plan(
    agent: Agent,
    width: int,
    height: int,
) -> GoalPlan | None:
    traversable = known_traversable_cells(agent)

    for candidate in ranked_goal_candidates(
        agent,
        width,
        height,
    ):
        plan = build_plan(
            kind=candidate.kind,
            targets=candidate.targets,
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
    return select_goal_plan(
        agent=agent,
        width=width,
        height=height,
    ) is not None


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