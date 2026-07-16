from dataclasses import dataclass

from src.configs.validation import (
    require_non_empty_text,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class EpisodicMemoryConfig:
    enabled: bool = False
    store_path: str = "artifacts/episodic_memory.jsonl"
    max_prior_episodes: int = 2_000
    max_store_episodes: int = 10_000
    require_world_signature_match: bool = True
    save_after_run: bool = True

    def __post_init__(self) -> None:
        require_non_empty_text(
            "episodic_memory.store_path",
            self.store_path,
        )
        require_positive(
            "episodic_memory.max_prior_episodes",
            self.max_prior_episodes,
        )
        require_positive(
            "episodic_memory.max_store_episodes",
            self.max_store_episodes,
        )
