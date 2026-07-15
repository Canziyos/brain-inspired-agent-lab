import torch

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.learning.outcome import (
    EVENT_TYPES,
    OutcomePrediction,
    RateRecurrentOutcomeModel,
)
from src.planning.imagination import (
    choose_imagined_action,
    expected_reward_from_prediction,
    imagine_actions,
)


def probabilities_for(
    event: EventType,
) -> tuple[float, ...]:
    return tuple(
        1.0 if candidate is event else 0.0
        for candidate in EVENT_TYPES
    )


def test_expected_reward_uses_predicted_event_probabilities() -> None:
    prediction = OutcomePrediction(
        energy_change=-2.0,
        health_change=0.0,
        curiosity_change=-5.0,
        event=EventType.DISCOVERED_MYSTERY,
        event_probabilities=probabilities_for(
            EventType.DISCOVERED_MYSTERY
        ),
    )

    assert expected_reward_from_prediction(
        prediction
    ) == 6.0


def test_imagination_evaluates_each_action_from_same_state() -> None:
    model = RateRecurrentOutcomeModel(
        neuron_count=6,
        neural_ticks=2,
    )
    agent = Agent(x=1, y=1)
    evaluations = [
        ActionEvaluation(
            action=Action.REST,
            policy_score=0.0,
            rationale="rest",
        ),
        ActionEvaluation(
            action=Action.MOVE_EAST,
            policy_score=1.0,
            rationale="move",
        ),
    ]

    initial_state = model.initial_neural_state(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state_copy = initial_state.clone()

    reward_weight = 0.5

    imagined = imagine_actions(
        agent=agent,
        evaluations=evaluations,
        model=model,
        neural_state=initial_state,
        reward_weight=reward_weight,
    )

    assert len(imagined) == 2
    assert torch.equal(initial_state, state_copy)

    for evaluation, imagined_action in zip(
        evaluations,
        imagined,
        strict=True,
    ):
        assert imagined_action.policy_score == evaluation.policy_score
        assert imagined_action.utility == (
            imagined_action.policy_score
            + reward_weight * imagined_action.expected_reward
        )

    assert choose_imagined_action(imagined) in imagined
