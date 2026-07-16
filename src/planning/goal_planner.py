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
from src.planning.goal_scoring import (
    ScoredGoalPlan,
    score_goal_plan,
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
    score: float | None = None

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


@dataclass(frozen=True, slots=True)
class GoalPreference:
    kind: GoalKind
    target: Position
    continuation_bonus: float
    switch_margin: float


@dataclass(frozen=True, slots=True)
class PreferredGoalChoice:
    plan: GoalPlan
    total: float


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


def build_scored_plans(
    candidate: GoalCandidate,
    agent: Agent,
    traversable: set[Position],
    width: int,
    height: int,
) -> tuple[ScoredGoalPlan, ...]:
    scored_plans: list[ScoredGoalPlan] = []

    for target in sorted(candidate.targets):
        plan = build_plan(
            kind=candidate.kind,
            targets=(target,),
            agent=agent,
            traversable=traversable,
            width=width,
            height=height,
        )

        if plan is None:
            continue

        scored_plans.append(
            score_goal_plan(
                plan=plan,
                agent=agent,
                width=width,
                height=height,
                motivation=candidate.priority,
            )
        )

    return tuple(scored_plans)


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
    preference: GoalPreference | None = None,
) -> GoalPlan | None:
    traversable = known_traversable_cells(agent)
    scored_plans: list[ScoredGoalPlan] = []

    for candidate in goal_candidates(
        agent,
        width,
        height,
    ):
        scored_plans.extend(
            build_scored_plans(
                candidate=candidate,
                agent=agent,
                traversable=traversable,
                width=width,
                height=height,
            )
        )

    if not scored_plans:
        return None

    choice = choose_preferred_goal(
        scored_plans=scored_plans,
        preference=preference,
    )

    return GoalPlan(
        kind=choice.plan.kind,
        target=choice.plan.target,
        path=choice.plan.path,
        score=choice.total,
    )


def choose_preferred_goal(
    scored_plans: Collection[ScoredGoalPlan],
    preference: GoalPreference | None,
) -> PreferredGoalChoice:
    best = max(
        scored_plans,
        key=lambda scored_plan: scored_plan.total,
    )

    if preference is None:
        return PreferredGoalChoice(
            plan=best.plan,
            total=best.total,
        )

    preferred = find_preferred_scored_goal(
        scored_plans=scored_plans,
        preference=preference,
    )

    if preferred is None:
        return PreferredGoalChoice(
            plan=best.plan,
            total=best.total,
        )

    preferred_total = preferred.total + preference.continuation_bonus

    if goals_match(best.plan, preference):
        return PreferredGoalChoice(
            plan=best.plan,
            total=preferred_total,
        )

    switch_threshold = preferred_total + preference.switch_margin

    if best.total <= switch_threshold:
        return PreferredGoalChoice(
            plan=preferred.plan,
            total=preferred_total,
        )

    return PreferredGoalChoice(
        plan=best.plan,
        total=best.total,
    )


def find_preferred_scored_goal(
    scored_plans: Collection[ScoredGoalPlan],
    preference: GoalPreference,
) -> ScoredGoalPlan | None:
    for scored_plan in scored_plans:
        if goals_match(
            plan=scored_plan.plan,
            preference=preference,
        ):
            return scored_plan

    return None


def goals_match(
    plan: GoalPlan,
    preference: GoalPreference,
) -> bool:
    return (
        plan.kind is preference.kind
        and plan.target == preference.target
    )


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
