from src.core.actions import Action
from src.core.agent import Agent
from src.core.world import FOOD
from src.learning.features import (
    STATE_ACTION_FEATURE_COUNT,
    encode_state_action,
)


def make_action(name: str, x: int, y: int) -> Action:
    return Action(
        kind=name,
        target_x=x,
        target_y=y,
    )


def test_encode_state_action_marks_food_target() -> None:
    agent = Agent(
        x=0,
        y=0,
        energy=50.0,
        health=80.0,
        curiosity=25.0,
    )

    features = encode_state_action(
        agent_state=agent.snapshot(),
        action=make_action("move", 1, 0),
        perceived_cell=FOOD,
    )

    assert len(features) == STATE_ACTION_FEATURE_COUNT
    assert features == (
        0.5,
        0.8,
        0.25,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_encode_state_action_marks_rest_and_current_cell_visited() -> None:
    agent = Agent(x=0, y=0)

    features = encode_state_action(
        agent_state=agent.snapshot(),
        action=make_action("rest", 0, 0),
        perceived_cell=None,
    )

    assert features[3] == 1.0
    assert features[-1] == 1.0
