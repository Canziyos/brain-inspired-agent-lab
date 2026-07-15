from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import CellType
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)


def evaluations_by_action(
    evaluations: tuple[ActionEvaluation, ...],
) -> dict[Action, ActionEvaluation]:
    return {
        evaluation.action: evaluation
        for evaluation in evaluations
    }


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

    by_action = evaluations_by_action(evaluations)

    assert by_action[Action.REST].policy_score == 25.0
    assert by_action[Action.MOVE_EAST].policy_score == 70.0
    assert by_action[Action.MOVE_WEST].policy_score == 24.0
    assert by_action[Action.MOVE_NORTH].policy_score == 15.0
    assert Action.MOVE_SOUTH not in by_action


def test_evaluate_actions_suppresses_rest_when_energy_is_safe() -> None:
    agent = Agent(
        x=1,
        y=1,
        energy=70.0,
    )

    evaluations = evaluate_actions(
        agent,
        [
            Observation(1, 0, CellType.EMPTY),
        ],
    )

    by_action = evaluations_by_action(evaluations)

    assert Action.REST not in by_action
    assert Action.MOVE_NORTH in by_action


def test_evaluate_actions_keeps_rest_when_energy_is_low() -> None:
    agent = Agent(
        x=1,
        y=1,
        energy=50.0,
    )

    evaluations = evaluate_actions(
        agent,
        [
            Observation(1, 0, CellType.EMPTY),
        ],
    )

    by_action = evaluations_by_action(evaluations)

    assert Action.REST in by_action
    assert Action.MOVE_NORTH in by_action


def test_evaluate_actions_keeps_rest_when_no_safe_move_exists() -> None:
    agent = Agent(
        x=1,
        y=1,
        energy=70.0,
    )

    evaluations = evaluate_actions(
        agent,
        [
            Observation(1, 0, CellType.DANGER),
        ],
    )

    by_action = evaluations_by_action(evaluations)

    assert tuple(by_action) == (Action.REST,)


def test_choose_action_returns_best_score() -> None:
    chosen = choose_action(
        evaluations=(
            ActionEvaluation(
                action=Action.REST,
                policy_score=1.0,
            ),
            ActionEvaluation(
                action=Action.MOVE_NORTH,
                policy_score=2.0,
            ),
        ),
    )

    assert chosen.action is Action.MOVE_NORTH
