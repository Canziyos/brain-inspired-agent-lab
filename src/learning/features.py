from typing import Final

from src.core.actions import Action
from src.core.perception import AgentState
from src.core.world import CellType
from src.learning.types import FeatureVector


STATE_ACTION_FEATURE_COUNT: Final[int] = 10


def encode_state_action(
    agent_state: AgentState,
    action: Action,
    target_cell: CellType | None,
) -> FeatureVector:
    cell = target_cell if action.is_move else None
    target = action.target_from((agent_state.x, agent_state.y))

    features = (
        agent_state.energy / 100.0,
        agent_state.health / 100.0,
        agent_state.curiosity / 100.0,

        1.0 if action.is_rest else 0.0,

        1.0 if cell is CellType.EMPTY else 0.0,
        1.0 if cell is CellType.FOOD else 0.0,
        1.0 if cell is CellType.DANGER else 0.0,
        1.0 if cell is CellType.MYSTERY else 0.0,
        1.0 if action.is_move and cell is None else 0.0,

        1.0 if action.is_move and target in agent_state.visited else 0.0,
    )

    if len(features) != STATE_ACTION_FEATURE_COUNT:
        raise AssertionError(
            "State-action feature count is inconsistent."
        )

    return features