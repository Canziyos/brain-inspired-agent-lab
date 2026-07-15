from dataclasses import dataclass

from src.configs.validation import require_non_negative


@dataclass(frozen=True, slots=True)
class ImaginationConfig:
    reward_weight: float = 0.5

    def __post_init__(self) -> None:
        require_non_negative(
            "imagination.reward_weight",
            self.reward_weight,
        )