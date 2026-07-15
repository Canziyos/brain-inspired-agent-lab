from src.core.actions import Action


ACTION_REST = 0
ACTION_UP = 1
ACTION_UP_RIGHT = 2
ACTION_RIGHT = 3
ACTION_DOWN_RIGHT = 4
ACTION_DOWN = 5
ACTION_DOWN_LEFT = 6
ACTION_LEFT = 7
ACTION_UP_LEFT = 8

ACTION_COUNT = 9

DISCRETE_TO_ACTION = {
    ACTION_REST: Action.REST,
    ACTION_UP: Action.MOVE_NORTH,
    ACTION_UP_RIGHT: Action.MOVE_NORTH_EAST,
    ACTION_RIGHT: Action.MOVE_EAST,
    ACTION_DOWN_RIGHT: Action.MOVE_SOUTH_EAST,
    ACTION_DOWN: Action.MOVE_SOUTH,
    ACTION_DOWN_LEFT: Action.MOVE_SOUTH_WEST,
    ACTION_LEFT: Action.MOVE_WEST,
    ACTION_UP_LEFT: Action.MOVE_NORTH_WEST,
}

ACTION_TO_DISCRETE = {
    action: index
    for index, action in DISCRETE_TO_ACTION.items()
}

MOVE_ACTIONS = tuple(
    action
    for action in DISCRETE_TO_ACTION.values()
    if action.is_move
)

MOVE_DELTAS = tuple(
    action.delta
    for action in MOVE_ACTIONS
)


def action_to_discrete(
    position: tuple[int, int],
    action: Action,
) -> int:
    del position

    try:
        return ACTION_TO_DISCRETE[action]
    except KeyError as exc:
        raise ValueError(
            "Only rest and one-cell movement actions can be "
            "mapped to the GridWorld action space."
        ) from exc


def discrete_to_action(action_index: int) -> Action:
    try:
        return DISCRETE_TO_ACTION[action_index]
    except KeyError as exc:
        raise ValueError(f"Invalid action index: {action_index}") from exc