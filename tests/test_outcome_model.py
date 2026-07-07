import torch

from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.core.perception import AgentState
from src.core.world import FOOD
from src.learning.outcome_features import (
    OUTCOME_FEATURE_COUNT,
    encode_outcome_features,
)
from src.learning.outcome_model import (
    RateRecurrentOutcomeModel,
    event_index,
)


def test_outcome_features_encode_direction_and_cell() -> None:
    state = AgentState(
        x=2,
        y=3,
        energy=70.0,
        health=100.0,
        curiosity=50.0,
        visited=frozenset({(2, 3)}),
    )

    features = encode_outcome_features(
        agent_state=state,
        action=Action(
            kind="move",
            target_x=3,
            target_y=3,
        ),
        perceived_cell=FOOD,
    )

    assert len(features) == OUTCOME_FEATURE_COUNT

    # Right action.
    assert features[5] == 1.0

    # Food target.
    assert features[10] == 1.0


def test_recurrent_model_exposes_activity_over_ticks() -> None:
    model = RateRecurrentOutcomeModel(
        neuron_count=6,
        neural_ticks=4,
    )

    features = torch.zeros(
        2,
        OUTCOME_FEATURE_COUNT,
    )

    (
        changes,
        logits,
        activity,
    ) = model(
        features,
        return_activity=True,
    )

    assert changes.shape == (2, 3)

    assert logits.shape == (
        2,
        len(tuple(EventType)),
    )

    assert activity.shape == (
        2,
        4,
        6,
    )


def test_event_indices_cover_every_event() -> None:
    indices = {
        event_index(event)
        for event in EventType
    }

    assert indices == set(
        range(len(tuple(EventType)))
    )

def test_recurrent_model_can_continue_from_external_state():
    model = RateRecurrentOutcomeModel(
        neuron_count=6,
        neural_ticks=4,
    )

    features = torch.ones(
        1,
        OUTCOME_FEATURE_COUNT,
    )

    state0 = model.initial_neural_state(
        batch_size=1,
        device=features.device,
        dtype=features.dtype,
    )

    changes1, logits1, state1 = model.forward_with_state(
        features,
        state0,
    )

    changes2, logits2, state2 = model.forward_with_state(
        features,
        state1,
    )

    assert state1.shape == (1, 6)
    assert state2.shape == (1, 6)

    assert changes1.shape == (1, 3)
    assert changes2.shape == (1, 3)

    assert logits1.shape[0] == 1
    assert logits2.shape[0] == 1

    assert not torch.equal(state0, state1)