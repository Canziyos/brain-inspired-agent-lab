from dataclasses import dataclass
from math import isfinite
from src.learning.types import FeatureVector, StateDelta

@dataclass(frozen=True, slots=True)
class RewardTrainingSample:
    features: FeatureVector
    reward: float

    def __post_init__(self) -> None:
        _validate_feature_vector(self.features)

        if not isfinite(self.reward):
            raise ValueError("reward must be finite")


@dataclass(frozen=True, slots=True)
class TransitionTrainingSample:
    features: FeatureVector
    state_delta: StateDelta
    event_index: int

    def __post_init__(self) -> None:
        _validate_feature_vector(self.features)

        if not all(isfinite(value) for value in self.state_delta):
            raise ValueError("state_delta must contain only finite numbers")

        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")


def _validate_feature_vector(features: FeatureVector) -> None:
    if not features:
        raise ValueError("features must not be empty")

    if not all(isfinite(value) for value in features):
        raise ValueError("features must contain only finite numbers")
