from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import (
    DANGER,
    EMPTY,
    FOOD,
    MYSTERY,
)
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
            Observation(2, 1, FOOD),
            Observation(1, 2, DANGER),
            Observation(0, 1, MYSTERY),
            Observation(1, 0, EMPTY),
        ],
    )

    by_target = {
        (
            evaluation.action.kind,
            evaluation.action.target_x,
            evaluation.action.target_y,
        ): evaluation
        for evaluation in evaluations
    }

    assert by_target[("rest", 1, 1)].score == 25.0
    assert by_target[("move", 2, 1)].score == 70.0
    assert by_target[("move", 0, 1)].score == 24.0
    assert by_target[("move", 1, 0)].score == 15.0
    assert ("move", 1, 2) not in by_target