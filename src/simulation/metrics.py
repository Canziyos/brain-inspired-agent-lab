from dataclasses import dataclass

   
@dataclass(slots=True)
class OutcomeModelMetrics:
    predicted_energy_change: float
    actual_energy_change: float

    predicted_health_change: float
    actual_health_change: float

    predicted_curiosity_change: float
    actual_curiosity_change: float

    predicted_event: str
    actual_event: str
    event_correct: bool

    state_mae: float

    total_loss: float | None
    state_loss: float | None
    event_loss: float | None

    final_neural_state: tuple[float, ...]
    
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
    outcome_model: OutcomeModelMetrics
 