from src.core.actions import Action
from src.core.perception import AgentState
from src.core.world import (
    DANGER,
    EMPTY,
    FOOD,
    MYSTERY,
)


OUTCOME_FEATURE_COUNT = 15


def encode_outcome_features(
    agent_state: AgentState,
    action: Action,
    perceived_cell: str | None,
) -> tuple[float, ...]:
    dx = action.target_x - agent_state.x
    dy = action.target_y - agent_state.y

    is_rest = action.kind == "rest"
    is_blocked = action.kind == "blocked"
    is_move = action.kind == "move"

    cell = perceived_cell if is_move else None

    target = (
        action.target_x,
        action.target_y,
    )

    features = (
        # Internal state
        agent_state.energy / 100.0,
        agent_state.health / 100.0,
        agent_state.curiosity / 100.0,

        # Action identity
        1.0 if is_rest else 0.0,
        1.0 if is_move and (dx, dy) == (0, -1) else 0.0,
        1.0 if is_move and (dx, dy) == (1, 0) else 0.0,
        1.0 if is_move and (dx, dy) == (0, 1) else 0.0,
        1.0 if is_move and (dx, dy) == (-1, 0) else 0.0,
        1.0 if is_blocked else 0.0,

        # Believed target type
        1.0 if cell == EMPTY else 0.0,
        1.0 if cell == FOOD else 0.0,
        1.0 if cell == DANGER else 0.0,
        1.0 if cell == MYSTERY else 0.0,
        1.0 if is_move and cell is None else 0.0,

        # Memory
        1.0 if target in agent_state.visited else 0.0,
    )

    if len(features) != OUTCOME_FEATURE_COUNT:
        raise AssertionError(
            "Outcome feature count is inconsistent."
        )

    return features