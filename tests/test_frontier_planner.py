from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import DANGER, EMPTY
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
            (1, 1): EMPTY,
            (2, 1): EMPTY,
            (3, 1): EMPTY,
            (4, 1): EMPTY,

            (1, 0): DANGER,
            (1, 2): DANGER,
            (2, 0): DANGER,
            (2, 2): DANGER,
            (3, 0): DANGER,
            (3, 2): DANGER,
            (4, 0): DANGER,
            (4, 2): DANGER,
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
            Observation(3, 1, EMPTY),
            Observation(4, 0, DANGER),
            Observation(4, 2, DANGER),
        ],
        plan=plan,
    )

    chosen = choose_action(evaluations)

    assert chosen.action.target_x == 3
    assert chosen.action.target_y == 1
    assert (
        chosen.reason
        == "follow frontier goal toward (1, 1)"
    )