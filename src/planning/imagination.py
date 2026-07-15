from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.learning.outcome import (
    EVENT_TYPES,
    OutcomePrediction,
    RateRecurrentOutcomeModel,
    predict_outcome_with_state,
)
from src.learning.outcome_features import encode_outcome_features
from src.learning.rewards import (
    HEALTH_REWARD_WEIGHT,
    event_reward_adjustment,
)
from src.learning.types import FeatureVector


@dataclass(frozen=True, slots=True)
class ImaginedAction:
    action: Action
    features: FeatureVector
    prediction: OutcomePrediction

    policy_score: float
    expected_reward: float
    utility: float

    next_neural_state: torch.Tensor


def expected_reward_from_prediction(
    prediction: OutcomePrediction,
) -> float:
    expected_event_reward = sum(
        probability * event_reward_adjustment(event)
        for event, probability in zip(
            EVENT_TYPES,
            prediction.event_probabilities,
            strict=True,
        )
    )

    return (
        prediction.energy_change
        + HEALTH_REWARD_WEIGHT * prediction.health_change
        + expected_event_reward
    )


def target_cell_for_action(
    agent: Agent,
    action: Action,
):
    target_position = action.target_from(agent.position)
    return agent.known_cells.get(target_position)


def imagine_actions(
    agent: Agent,
    evaluations: Sequence[ActionEvaluation],
    model: RateRecurrentOutcomeModel,
    neural_state: torch.Tensor,
    reward_weight: float,
) -> tuple[ImaginedAction, ...]:
    if not evaluations:
        raise ValueError(
            "Cannot imagine outcomes without candidate actions"
        )

    if reward_weight < 0.0:
        raise ValueError("reward_weight must be non-negative")

    agent_state = agent.snapshot()
    imagined: list[ImaginedAction] = []

    for evaluation in evaluations:
        action = evaluation.action

        features = encode_outcome_features(
            agent_state=agent_state,
            action=action,
            target_cell=target_cell_for_action(
                agent=agent,
                action=action,
            ),
        )

        prediction, next_neural_state = predict_outcome_with_state(
            model=model,
            features=features,
            neural_state=neural_state,
        )

        expected_reward = expected_reward_from_prediction(
            prediction
        )

        utility = (
            evaluation.policy_score
            + reward_weight * expected_reward
        )

        imagined.append(
            ImaginedAction(
                action=action,
                features=features,
                prediction=prediction,
                policy_score=evaluation.policy_score,
                expected_reward=expected_reward,
                utility=utility,
                next_neural_state=next_neural_state.detach(),
            )
        )

    return tuple(imagined)


def choose_imagined_action(
    imagined_actions: Sequence[ImaginedAction],
) -> ImaginedAction:
    if not imagined_actions:
        raise ValueError(
            "Cannot choose from empty imagined actions"
        )

    return max(
        imagined_actions,
        key=lambda imagined: imagined.utility,
    )


def find_imagined_action(
    chosen_action: Action,
    imagined_actions: Sequence[ImaginedAction],
) -> ImaginedAction:
    for imagined in imagined_actions:
        if imagined.action is chosen_action:
            return imagined

    raise ValueError(
        "Chosen action was not found in imagined actions"
    )