from dataclasses import dataclass
from enum import Enum


Position = tuple[int, int]


class Action(Enum):
    REST = (0, 0)

    MOVE_NORTH = (0, -1)
    MOVE_NORTH_EAST = (1, -1)
    MOVE_EAST = (1, 0)
    MOVE_SOUTH_EAST = (1, 1)
    MOVE_SOUTH = (0, 1)
    MOVE_SOUTH_WEST = (-1, 1)
    MOVE_WEST = (-1, 0)
    MOVE_NORTH_WEST = (-1, -1)

    @property
    def delta(self) -> Position:
        dx, dy = self.value
        return dx, dy

    @property
    def is_rest(self) -> bool:
        return self is Action.REST

    @property
    def is_move(self) -> bool:
        return not self.is_rest

    @property
    def is_diagonal(self) -> bool:
        dx, dy = self.delta
        return dx != 0 and dy != 0

    def target_from(self, position: Position) -> Position:
        x, y = position
        dx, dy = self.delta
        return x + dx, y + dy

    @classmethod
    def from_delta(cls, dx: int, dy: int) -> "Action":
        for action in cls:
            if action.delta == (dx, dy):
                return action

        raise ValueError(f"No one-step action for delta {(dx, dy)}")

    @classmethod
    def from_positions(
        cls,
        start: Position,
        target: Position,
    ) -> "Action":
        sx, sy = start
        tx, ty = target
        return cls.from_delta(tx - sx, ty - sy)


MOVE_ACTIONS = tuple(
    action for action in Action if action.is_move
)

MOVE_DELTAS = tuple(
    action.delta for action in MOVE_ACTIONS
)


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: Action
    policy_score: float
    rationale: str = ""