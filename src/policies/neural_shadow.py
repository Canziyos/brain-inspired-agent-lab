from collections.abc import Sequence
from dataclasses import dataclass

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.learning.features import encode_state_action
from src.learning.reward_network import (
    ImmediateRewardNetwork,
    predict_reward,
)
from src.learning.types import FeatureVector


@dataclass(frozen=True, slots=True)
class NetworkActionPrediction:
    action: Action
    features: FeatureVector
    predicted_reward: float


def predict_network_actions(
    agent: Agent,
    evaluations: Sequence[ActionEvaluation],
    model: ImmediateRewardNetwork,
) -> list[NetworkActionPrediction]:
    predictions: list[NetworkActionPrediction] = []
    agent_state = agent.snapshot()

    for evaluation in evaluations:
        action = evaluation.action
        target = action.target_from(agent.position)

        target_cell = (
            agent.known_cells.get(target)
            if action.is_move
            else None
        )

        features = encode_state_action(
            agent_state=agent_state,
            action=action,
            target_cell=target_cell,
        )

        predicted_reward = predict_reward(
            model=model,
            features=features,
        )

        predictions.append(
            NetworkActionPrediction(
                action=action,
                features=features,
                predicted_reward=predicted_reward,
            )
        )

    return predictions


def choose_network_action(
    predictions: Sequence[NetworkActionPrediction],
) -> NetworkActionPrediction:
    if not predictions:
        raise ValueError(
            "Cannot choose from empty network predictions"
        )

    return max(
        predictions,
        key=lambda prediction: prediction.predicted_reward,
    )


def find_action_prediction(
    chosen_action: Action,
    predictions: Sequence[NetworkActionPrediction],
) -> NetworkActionPrediction:
    for prediction in predictions:
        if prediction.action is chosen_action:
            return prediction

    raise ValueError(
        "Chosen action was not found in network predictions"
    )