from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.agent import Agent
from src.core.dynamics import MOVE_ENERGY_COST
from src.core.motivation import MEDIUM_ENERGY
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

ENERGY_RESERVE_TARGET = 40.0
ENERGY_RESERVE_SHORTFALL_COST = 1.0
FOOD_ENERGY_RISK_SCALE = 0.25

FOOD_URGENCY_BONUS_PER_ENERGY = 1.2


@dataclass(frozen=True, slots=True)
class GoalScore:
    motivation: float
    information_gain: float
    energy_bonus: float
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
    energy_bonus = goal_energy_bonus(
        plan=plan,
        agent=agent,
    )
    distance_cost = distance * DISTANCE_COST_PER_STEP
    danger_risk = known_danger_risk(
        plan=plan,
        agent=agent,
        width=width,
        height=height,
    )
    energy_risk = travel_energy_risk(
        plan=plan,
        agent=agent,
        distance=distance,
    )

    total = (
        motivation
        + information_gain
        + energy_bonus
        - distance_cost
        - danger_risk
        - energy_risk
    )

    return ScoredGoalPlan(
        plan=plan,
        score=GoalScore(
            motivation=motivation,
            information_gain=information_gain,
            energy_bonus=energy_bonus,
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


def is_food_goal(plan: GoalPlan) -> bool:
    return goal_kind_value(plan.kind) == "food"


def goal_energy_bonus(
    plan: GoalPlan,
    agent: Agent,
) -> float:
    if not is_food_goal(plan):
        return 0.0

    energy_shortfall = max(0.0, MEDIUM_ENERGY - agent.energy)
    return energy_shortfall * FOOD_URGENCY_BONUS_PER_ENERGY


def travel_energy_risk(
    plan: GoalPlan,
    agent: Agent,
    distance: int,
) -> float:
    expected_energy_cost = distance * MOVE_ENERGY_COST
    projected_energy = agent.energy - expected_energy_cost
    reserve_shortfall = max(
        0.0,
        ENERGY_RESERVE_TARGET - projected_energy,
    )

    risk = reserve_shortfall * ENERGY_RESERVE_SHORTFALL_COST

    if is_food_goal(plan):
        return risk * FOOD_ENERGY_RISK_SCALE

    return risk


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
