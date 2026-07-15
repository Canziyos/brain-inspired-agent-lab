from typing import Final

from src.core.actions import Action
from src.core.perception import AgentState
from src.core.world import CellType
from src.learning.types import FeatureVector


OUTCOME_FEATURE_COUNT: Final[int] = 18

ACTION_IDENTITY_FEATURES: Final[tuple[Action, ...]] = (
    Action.REST,
    Action.MOVE_NORTH,
    Action.MOVE_NORTH_EAST,
    Action.MOVE_EAST,
    Action.MOVE_SOUTH_EAST,
    Action.MOVE_SOUTH,
    Action.MOVE_SOUTH_WEST,
    Action.MOVE_WEST,
    Action.MOVE_NORTH_WEST,
)


def encode_outcome_features(
    agent_state: AgentState,
    action: Action,
    target_cell: CellType | None,
) -> FeatureVector:
    is_move = action.is_move
    cell = target_cell if is_move else None
    target = action.target_from((agent_state.x, agent_state.y))

    features = (
        # Internal state
        agent_state.energy / 100.0,
        agent_state.health / 100.0,
        agent_state.curiosity / 100.0,

        # Action identity
        *_encode_action_identity(action),

        # Believed target type
        1.0 if cell is CellType.EMPTY else 0.0,
        1.0 if cell is CellType.FOOD else 0.0,
        1.0 if cell is CellType.DANGER else 0.0,
        1.0 if cell is CellType.MYSTERY else 0.0,
        1.0 if is_move and cell is None else 0.0,

        # Memory
        1.0 if is_move and target in agent_state.visited else 0.0,
    )

    if len(features) != OUTCOME_FEATURE_COUNT:
        raise AssertionError(
            "Outcome feature count is inconsistent."
        )

    return features


def _encode_action_identity(action: Action) -> FeatureVector:
    return tuple(
        1.0 if action is candidate else 0.0
        for candidate in ACTION_IDENTITY_FEATURES
    )