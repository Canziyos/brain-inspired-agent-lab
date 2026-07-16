from dataclasses import dataclass
from typing import TypeAlias

from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.core.world import CellType


Position: TypeAlias = tuple[int, int]
GridSnapshot: TypeAlias = tuple[tuple[CellType, ...], ...]


@dataclass(frozen=True, slots=True)
class OutcomeModelMetrics:
    predicted_energy_change: float
    actual_energy_change: float

    predicted_health_change: float
    actual_health_change: float

    predicted_curiosity_change: float
    actual_curiosity_change: float

    predicted_event: EventType
    actual_event: EventType
    event_correct: bool

    state_mae: float
    reset_state_mae: float
    persistent_better_than_reset: bool

    total_loss: float | None
    state_loss: float | None
    event_loss: float | None

    neural_state_mean: float
    neural_state_std: float
    neural_state_min: float
    neural_state_max: float


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    position: Position

    energy: float
    health: float
    curiosity: float

    reward: float
    predicted_reward: float
    loss: float | None

    event: EventType
    visited_count: int
    known_cell_count: int

    choices_agree: bool

    grid_snapshot: GridSnapshot

    rule_action: Action
    action_reason: str
    network_action: Action

    imagination_action: Action

    imagination_expected_reward: float
    imagination_utility: float

    rule_imagined_reward: float
    rule_imagined_utility: float

    imagination_agrees: bool

    termination_reason: str | None
    goal_kind: str | None
    goal_target: Position | None

    memory_goal_age: int
    memory_goal_switch_count: int
    memory_recent_revisit_count: int
    memory_stuck_counter: int
    memory_recent_rest_count: int
    memory_energy_trend: float
    memory_last_food_position: Position | None
    memory_last_mystery_position: Position | None
    memory_last_danger_position: Position | None

    outcome_model: OutcomeModelMetrics | None
