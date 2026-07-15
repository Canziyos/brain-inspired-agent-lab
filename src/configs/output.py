from dataclasses import dataclass

from src.configs.validation import (
    require_non_empty_text,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    show_animation: bool = False
    show_plots: bool = False

    animation_interval_ms: int = 250

    save_run_outputs: bool = False
    output_root: str = "runs"

    def __post_init__(self) -> None:
        require_positive(
            "output.animation_interval_ms",
            self.animation_interval_ms,
        )
        require_non_empty_text("output.output_root", self.output_root)