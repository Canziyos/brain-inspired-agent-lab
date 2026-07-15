from dataclasses import dataclass

from src.configs.validation import (
    require_positive,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class OutcomeConfig:
    learning_rate: float = 0.003
    neuron_count: int = 24
    neural_ticks: int = 8
    leak: float = 0.35
    sequence_length: int = 8
    sequence_batch_size: int = 4

    def __post_init__(self) -> None:
        require_positive("outcome.learning_rate", self.learning_rate)
        require_positive("outcome.neuron_count", self.neuron_count)
        require_positive("outcome.neural_ticks", self.neural_ticks)
        require_probability("outcome.leak", self.leak)
        require_positive("outcome.sequence_length", self.sequence_length)
        require_positive(
            "outcome.sequence_batch_size",
            self.sequence_batch_size,
        )