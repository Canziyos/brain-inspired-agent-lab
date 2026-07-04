from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    target_x: int
    target_y: int


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: Action
    score: float
    reason: str

