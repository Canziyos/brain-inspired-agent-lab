from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardSample:
    features: tuple[float, ...]
    reward: float


@dataclass(frozen=True, slots=True)
class OutcomeSample:
    features: tuple[float, ...]
    state_changes: tuple[float, float, float]
    event_index: int
    neural_state: tuple[float, ...]