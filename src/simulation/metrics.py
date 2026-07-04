from dataclasses import dataclass


@dataclass(slots=True)
class StepMetrics:
    step: int
    position: tuple[int, int]

    energy: float
    health: float
    curiosity: float

    reward: float
    predicted_reward: float
    loss: float | None

    event: str
    visited_count: int
    known_cell_count: int

    choices_agree: bool

    grid_snapshot: tuple[tuple[str, ...], ...]

    rule_action: str
    action_reason: str
    network_action: str

    termination_reason: str | None
    goal_kind: str | None
    goal_target: tuple[int, int] | None