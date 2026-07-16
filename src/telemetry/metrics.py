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

    episodic_action: Action | None
    episodic_expected_reward: float
    episodic_confidence: float
    episodic_match_count: int
    episodic_best_event: EventType | None
    episodic_risk_hit_danger: float
    episodic_agrees_with_rule: bool
    episodic_agrees_with_imagination: bool
    episodic_rationale: str

    termination_reason: str | None
    goal_kind: str | None
    goal_target: Position | None

    memory_goal_id: str | None
    memory_goal_age: int
    memory_goal_switch_count: int
    memory_target_switch_count: int
    memory_frontier_target_switch_count: int
    memory_frontier_semantic_switch_count: int
    memory_recent_revisit_count: int
    memory_stuck_counter: int
    memory_recent_rest_count: int
    memory_energy_trend: float
    memory_last_food_position: Position | None
    memory_last_mystery_position: Position | None
    memory_last_danger_position: Position | None

    coverage_total_world_cells: int
    coverage_seen_cell_count: int
    coverage_visited_cell_count: int
    coverage_unseen_cell_count: int
    coverage_seen_ratio: float
    coverage_visited_ratio: float
    coverage_frontier_count: int
    coverage_reachable_frontier_count: int
    coverage_unreachable_frontier_count: int
    coverage_frontier_cluster_count: int
    coverage_reachable_frontier_cluster_count: int
    coverage_current_frontier_cluster_id: str | None
    coverage_newly_seen_count: int
    coverage_newly_visited_count: int
    coverage_first_full_seen_step: int | None
    coverage_first_full_visited_step: int | None

    outcome_model: OutcomeModelMetrics | None
