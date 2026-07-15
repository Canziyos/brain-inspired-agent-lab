from src.core.agent import Agent
from src.core.world import CellType
from src.planning.goal_planner import (
    GoalKind,
    is_task_complete,
    select_goal_plan,
)


def test_low_energy_agent_prioritizes_known_food() -> None:
    agent = Agent(
        x=0,
        y=0,
        energy=50.0,
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
    assert plan.next_step == (1, 0)


def test_high_energy_agent_prioritizes_mystery() -> None:
    agent = Agent(
        x=0,
        y=0,
        energy=95.0,
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
    assert plan.kind is GoalKind.MYSTERY
    assert plan.target == (0, 1)


def test_frontier_is_selected_without_known_objects() -> None:
    agent = Agent(x=0, y=0)

    agent.known_cells.update(
        {
            (0, 0): CellType.EMPTY,
            (1, 0): CellType.EMPTY,
        }
    )

    plan = select_goal_plan(
        agent=agent,
        width=3,
        height=1,
    )

    assert plan is not None
    assert plan.kind is GoalKind.FRONTIER
    assert plan.target == (1, 0)
    assert plan.next_step == (1, 0)


def test_task_is_complete_when_no_reachable_goals_remain() -> None:
    agent = Agent(x=0, y=0)

    agent.known_cells.update(
        {
            (0, 0): CellType.EMPTY,
            (1, 0): CellType.EMPTY,
        }
    )

    assert is_task_complete(
        agent=agent,
        width=2,
        height=1,
    )
