from dataclasses import dataclass, field

from src.core.dynamics_types import ActionOutcome
from src.core.perception import AgentState, Observation
from src.core.world import CellType, World


Position = tuple[int, int]

MIN_STAT = 0.0
MAX_STAT = 100.0

DEFAULT_ENERGY = 70.0
DEFAULT_HEALTH = 100.0
DEFAULT_CURIOSITY = 50.0


@dataclass(slots=True)
class Agent:
    x: int
    y: int
    energy: float = DEFAULT_ENERGY
    health: float = DEFAULT_HEALTH
    curiosity: float = DEFAULT_CURIOSITY

    known_cells: dict[Position, CellType] = field(
        default_factory=dict
    )
    visited: set[Position] = field(default_factory=set)
    experiences: list[ActionOutcome] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(
                "Agent position must be non-negative"
            )

        self.clamp_state()
        self.visited.add(self.position)

    @property
    def position(self) -> Position:
        return self.x, self.y

    def clamp_state(self) -> None:
        self.energy = clamp_stat(self.energy)
        self.health = clamp_stat(self.health)
        self.curiosity = clamp_stat(self.curiosity)

    def sense(self, world: World) -> list[Observation]:
        return [
            Observation(x=x, y=y, cell=cell)
            for x, y, cell in world.neighbors(self.x, self.y)
        ]

    def observe(
        self,
        observations: list[Observation],
    ) -> None:
        for observation in observations:
            self.remember_cell(
                position=(observation.x, observation.y),
                cell=observation.cell,
            )

    def remember_cell(
        self,
        position: Position,
        cell: CellType,
    ) -> None:
        self.known_cells[position] = cell

    def snapshot(self) -> AgentState:
        return AgentState(
            x=self.x,
            y=self.y,
            energy=self.energy,
            health=self.health,
            curiosity=self.curiosity,
            visited=frozenset(self.visited),
        )

    def move_to(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.visited.add(self.position)

    def record_experience(
        self,
        outcome: ActionOutcome,
    ) -> None:
        self.experiences.append(outcome)

    def is_alive(self) -> bool:
        return self.energy > MIN_STAT and self.health > MIN_STAT


def clamp_stat(value: float) -> float:
    return max(MIN_STAT, min(MAX_STAT, value))