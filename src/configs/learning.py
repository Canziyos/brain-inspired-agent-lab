from dataclasses import dataclass

from src.configs.validation import require_positive


@dataclass(frozen=True, slots=True)
class LearningConfig:
    learning_rate: float = 0.01
    batch_size: int = 8

    def __post_init__(self) -> None:
        require_positive("learning.learning_rate", self.learning_rate)
        require_positive("learning.batch_size", self.batch_size)