from dataclasses import dataclass

from src.configs.validation import require_positive


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_steps: int = 500

    random_seed: int = 7
    torch_seed: int = 7

    verbose: bool = False

    def __post_init__(self) -> None:
        require_positive("runtime.max_steps", self.max_steps)