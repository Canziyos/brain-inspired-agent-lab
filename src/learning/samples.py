from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardSample:
    features: tuple[float, ...]
    reward: float