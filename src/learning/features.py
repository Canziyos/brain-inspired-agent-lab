from src.core.actions import Action
from src.core.perception import AgentState
from src.core.world import (
    DANGER,
    EMPTY,
    FOOD,
    MYSTERY,
)


STATE_ACTION_FEATURE_COUNT = 9


def encode_state_action(
    agent_state: AgentState,
    action: Action,
    perceived_cell: str | None,
) -> tuple[float, ...]:
    cell = (
        None
        if action.kind == "rest"
        else perceived_cell
    )

    target = (
        action.target_x,
        action.target_y,
    )

    return (
        agent_state.energy / 100.0,
        agent_state.health / 100.0,
        agent_state.curiosity / 100.0,
        1.0 if action.kind == "rest" else 0.0,
        1.0 if cell == EMPTY else 0.0,
        1.0 if cell == FOOD else 0.0,
        1.0 if cell == DANGER else 0.0,
        1.0 if cell == MYSTERY else 0.0,
        1.0 if target in agent_state.visited else 0.0,
    )
