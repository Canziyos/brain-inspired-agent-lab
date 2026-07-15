from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.agent import Agent
from src.core.dynamics import MOVE_ENERGY_COST
from src.core.world import CellType
from src.planning.frontier_planner import Position, neighbor_positions

if TYPE_CHECKING:
    from src.planning.goal_planner import GoalPlan


GOAL_INFORMATION_GAIN = {
    "food": 4.0,
    "mystery": 22.0,
    "frontier": 14.0,
}

DISTANCE_COST_PER_STEP = 4.0
TARGET_DANGER_ADJACENCY_COST = 12.0
PATH_DANGER_ADJACENCY_COST = 3.0
ENERGY_SHORTFALL_COST = 2.5


@dataclass(frozen=True, slots=True)
class GoalScore:
    motivation: float
    information_gain: float
    distance_cost: float
    danger_risk: float
    energy_risk: float
    total: float


@dataclass(frozen=True, slots=True)
class ScoredGoalPlan:
    plan: GoalPlan
    score: GoalScore

    @property
    def total(self) -> float:
        return self.score.total


def score_goal_plan(
    plan: GoalPlan,
    agent: Agent,
    width: int,
    height: int,
    motivation: float,
) -> ScoredGoalPlan:
    distance = path_distance(plan.path)
    information_gain = goal_information_gain(plan)
    distance_cost = distance * DISTANCE_COST_PER_STEP
    danger_risk = known_danger_risk(
        plan=plan,
        agent=agent,
        width=width,
        height=height,
    )
    energy_risk = travel_energy_risk(
        agent=agent,
        distance=distance,
    )

    total = (
        motivation
        + information_gain
        - distance_cost
        - danger_risk
        - energy_risk
    )

    return ScoredGoalPlan(
        plan=plan,
        score=GoalScore(
            motivation=motivation,
            information_gain=information_gain,
            distance_cost=distance_cost,
            danger_risk=danger_risk,
            energy_risk=energy_risk,
            total=total,
        ),
    )


def path_distance(path: tuple[Position, ...]) -> int:
    return max(0, len(path) - 1)


def goal_information_gain(plan: GoalPlan) -> float:
    return GOAL_INFORMATION_GAIN.get(
        goal_kind_value(plan.kind),
        0.0,
    )


def goal_kind_value(kind: object) -> str:
    value = getattr(kind, "value", None)

    if isinstance(value, str):
        return value

    return str(kind)


def travel_energy_risk(
    agent: Agent,
    distance: int,
) -> float:
    expected_energy_cost = distance * MOVE_ENERGY_COST
    shortfall = max(0.0, expected_energy_cost - agent.energy)
    return shortfall * ENERGY_SHORTFALL_COST


def known_danger_risk(
    plan: GoalPlan,
    agent: Agent,
    width: int,
    height: int,
) -> float:
    known_dangers = {
        position
        for position, cell in agent.known_cells.items()
        if cell is CellType.DANGER
    }

    if not known_dangers:
        return 0.0

    target_risk = adjacent_danger_count(
        position=plan.target,
        known_dangers=known_dangers,
        width=width,
        height=height,
    ) * TARGET_DANGER_ADJACENCY_COST

    path_risk = sum(
        adjacent_danger_count(
            position=position,
            known_dangers=known_dangers,
            width=width,
            height=height,
        )
        for position in set(plan.path[1:])
    ) * PATH_DANGER_ADJACENCY_COST

    return target_risk + path_risk


def adjacent_danger_count(
    position: Position,
    known_dangers: set[Position],
    width: int,
    height: int,
) -> int:
    return sum(
        1
        for adjacent in neighbor_positions(
            position,
            width,
            height,
        )
        if adjacent in known_dangers
    )
