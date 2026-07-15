from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import CellType
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)


def test_evaluate_actions_scores_observed_options() -> None:
    agent = Agent(
        x=1,
        y=1,
        energy=50.0,
        curiosity=40.0,
    )

    evaluations = evaluate_actions(
        agent,
        [
            Observation(2, 1, CellType.FOOD),
            Observation(1, 2, CellType.DANGER),
            Observation(0, 1, CellType.MYSTERY),
            Observation(1, 0, CellType.EMPTY),
        ],
    )

    by_target = {
        (
            evaluation.action,
        ): evaluation
        for evaluation in evaluations
    }

    assert by_target[(Action.REST,)].policy_score == 25.0
    assert by_target[(Action.MOVE_EAST,)].policy_score == 70.0
    assert by_target[(Action.MOVE_WEST,)].policy_score == 24.0
    assert by_target[(Action.MOVE_NORTH,)].policy_score == 15.0
    assert (Action.MOVE_SOUTH,) not in by_target
