from src.core.actions import Action
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import CellType
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
)
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)


def create_corridor_agent() -> Agent:
    agent = Agent(
        x=4,
        y=1,
        energy=80.0,
    )

    agent.visited.update(
        {
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        }
    )

    agent.known_cells.update(
        {
            (1, 1): CellType.EMPTY,
            (2, 1): CellType.EMPTY,
            (3, 1): CellType.EMPTY,
            (4, 1): CellType.EMPTY,

            (1, 0): CellType.DANGER,
            (1, 2): CellType.DANGER,
            (2, 0): CellType.DANGER,
            (2, 2): CellType.DANGER,
            (3, 0): CellType.DANGER,
            (3, 2): CellType.DANGER,
            (4, 0): CellType.DANGER,
            (4, 2): CellType.DANGER,
        }
    )

    return agent


def test_policy_prefers_frontier_travel_over_revisit() -> None:
    agent = create_corridor_agent()

    plan = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 1),
        path=(
            (4, 1),
            (3, 1),
            (2, 1),
            (1, 1),
        ),
    )

    evaluations = evaluate_actions(
        agent,
        observations=[
            Observation(3, 1, CellType.EMPTY),
            Observation(4, 0, CellType.DANGER),
            Observation(4, 2, CellType.DANGER),
        ],
        plan=plan,
    )

    chosen = choose_action(evaluations)

    assert chosen.action is Action.MOVE_WEST
    assert (
        chosen.rationale
        == "follow frontier goal toward (1, 1)"
    )
