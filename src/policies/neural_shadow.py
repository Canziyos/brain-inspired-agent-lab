from dataclasses import dataclass

from src.core.agent import Agent
from src.core.actions import Action, ActionEvaluation
from src.learning.features import encode_state_action
from src.learning.reward_network import (
    ImmediateRewardNetwork,
    predict_reward,
)


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    action: Action
    features: tuple[float, ...]
    predicted_reward: float


def predict_actions(
    agent: Agent,
    evaluations: list[ActionEvaluation],
    model: ImmediateRewardNetwork,
) -> list[ActionPrediction]:
    predictions: list[ActionPrediction] = []
    agent_state = agent.snapshot()

    for evaluation in evaluations:
        action = evaluation.action
        perceived_cell = agent.known_cells.get(
            (action.target_x, action.target_y)
        )

        features = encode_state_action(
            agent_state=agent_state,
            action=action,
            perceived_cell=perceived_cell,
        )

        predicted_reward = predict_reward(
            model=model,
            features=features,
        )

        predictions.append(
            ActionPrediction(
                action=action,
                features=features,
                predicted_reward=predicted_reward,
            )
        )

    return predictions


def choose_network_action(
    predictions: list[ActionPrediction],
) -> ActionPrediction:
    if not predictions:
        raise ValueError(
            "Cannot choose from empty network predictions"
        )

    return max(
        predictions,
        key=lambda prediction: prediction.predicted_reward,
    )


def actions_match(
    first: Action,
    second: Action,
) -> bool:
    return (
        first.kind == second.kind
        and first.target_x == second.target_x
        and first.target_y == second.target_y
    )


def find_action_prediction(
    chosen_action: Action,
    predictions: list[ActionPrediction],
) -> ActionPrediction:
    for prediction in predictions:
        if prediction.action == chosen_action:
            return prediction

    raise ValueError(
        "Chosen action was not found in network predictions"
    )
