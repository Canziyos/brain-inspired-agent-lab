from dataclasses import dataclass, field

from src.configs.episodic_memory import EpisodicMemoryConfig
from src.configs.imagination import ImaginationConfig
from src.configs.learning import LearningConfig
from src.configs.outcome import OutcomeConfig
from src.configs.output import OutputConfig
from src.configs.runtime import RuntimeConfig
from src.configs.world import WorldConfig


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    imagination: ImaginationConfig = field(
        default_factory=ImaginationConfig
    )
    episodic_memory: EpisodicMemoryConfig = field(
        default_factory=EpisodicMemoryConfig
    )
