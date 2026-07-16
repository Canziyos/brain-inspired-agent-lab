from src.core.agent import Agent
from src.core.world import CellType
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
    select_goal_plan,
)
from src.planning.goal_scoring import score_goal_plan


def test_near_frontier_can_beat_far_mystery() -> None:
    agent = Agent(x=0, y=0)

    agent.known_cells.update(
        {
            (0, 0): CellType.EMPTY,
            (1, 0): CellType.EMPTY,
        }
    )

    for x in range(0, 12):
        agent.known_cells[(x, 1)] = CellType.EMPTY

    agent.known_cells[(12, 1)] = CellType.MYSTERY

    plan = select_goal_plan(
        agent=agent,
        width=14,
        height=3,
    )

    assert plan is not None
    assert plan.kind is GoalKind.FRONTIER
    assert len(plan.path) == 2


def test_low_energy_agent_still_prioritizes_food() -> None:
    agent = Agent(
        x=0,
        y=0,
        energy=25.0,
    )

    agent.known_cells.update(
        {
            (0, 0): CellType.EMPTY,
            (1, 0): CellType.FOOD,
            (0, 1): CellType.MYSTERY,
        }
    )

    plan = select_goal_plan(
        agent=agent,
        width=2,
        height=2,
    )

    assert plan is not None
    assert plan.kind is GoalKind.FOOD
    assert plan.target == (1, 0)


def test_known_danger_near_target_reduces_goal_score() -> None:
    agent = Agent(x=0, y=0)
    agent.known_cells.update(
        {
            (0, 0): CellType.EMPTY,
            (0, 1): CellType.EMPTY,
            (1, 0): CellType.EMPTY,
            (2, 0): CellType.DANGER,
        }
    )

    safe_plan = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(0, 1),
        path=((0, 0), (0, 1)),
    )
    risky_plan = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        path=((0, 0), (1, 0)),
    )

    safe_score = score_goal_plan(
        plan=safe_plan,
        agent=agent,
        width=3,
        height=2,
        motivation=14.0,
    )
    risky_score = score_goal_plan(
        plan=risky_plan,
        agent=agent,
        width=3,
        height=2,
        motivation=14.0,
    )

    assert safe_score.score.danger_risk < risky_score.score.danger_risk
    assert safe_score.total > risky_score.total


def test_far_goal_can_still_win_when_motivation_is_high() -> None:
    agent = Agent(x=0, y=0)

    near_plan = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        path=((0, 0), (1, 0)),
    )
    far_plan = GoalPlan(
        kind=GoalKind.MYSTERY,
        target=(5, 0),
        path=(
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
        ),
    )

    near_score = score_goal_plan(
        plan=near_plan,
        agent=agent,
        width=6,
        height=1,
        motivation=14.0,
    )
    far_score = score_goal_plan(
        plan=far_plan,
        agent=agent,
        width=6,
        height=1,
        motivation=120.0,
    )

    assert far_score.total > near_score.total


def test_low_energy_non_food_plan_gets_energy_risk() -> None:
    agent = Agent(
        x=0,
        y=0,
        energy=31.0,
    )

    plan = GoalPlan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        path=((0, 0), (1, 0), (2, 0)),
    )

    scored = score_goal_plan(
        plan=plan,
        agent=agent,
        width=3,
        height=1,
        motivation=30.0,
    )

    assert scored.score.energy_risk > 0.0
    assert scored.total < (
        scored.score.motivation
        + scored.score.information_gain
        - scored.score.distance_cost
    )


def test_low_energy_food_plan_gets_urgency_bonus() -> None:
    hungry_agent = Agent(
        x=0,
        y=0,
        energy=31.0,
    )
    full_agent = Agent(
        x=0,
        y=0,
        energy=90.0,
    )

    plan = GoalPlan(
        kind=GoalKind.FOOD,
        target=(1, 0),
        path=((0, 0), (1, 0)),
    )

    hungry_score = score_goal_plan(
        plan=plan,
        agent=hungry_agent,
        width=2,
        height=1,
        motivation=40.0,
    )
    full_score = score_goal_plan(
        plan=plan,
        agent=full_agent,
        width=2,
        height=1,
        motivation=40.0,
    )

    assert hungry_score.score.energy_bonus > 0.0
    assert full_score.score.energy_bonus == 0.0
    assert hungry_score.total > full_score.total
