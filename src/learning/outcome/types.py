from dataclasses import dataclass

from src.core.dynamics_types import EventType


@dataclass(frozen=True, slots=True)
class OutcomePrediction:
    energy_change: float
    health_change: float
    curiosity_change: float

    event: EventType
    event_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OutcomeTrainingResult:
    total_loss: float
    state_loss: float
    event_loss: float