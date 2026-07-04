from dataclasses import dataclass, field

from src.core.dynamics_types import ActionOutcome, EventType
from src.core.perception import AgentState, Observation
from src.core.world import World


@dataclass(slots=True)
class Experience:
    position: tuple[int, int]
    event: EventType
    energy_change: float
    health_change: float
    curiosity_change: float


@dataclass(slots=True)
class Agent:
    x: int
    y: int
    energy: float = 70.0
    health: float = 100.0
    curiosity: float = 50.0

    known_cells: dict[tuple[int, int], str] = field(
        default_factory=dict
    )
    visited: set[tuple[int, int]] = field(default_factory=set)
    experiences: list[Experience] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Baby Vice cannot be born outside reality")

        self.energy = max(0.0, min(100.0, self.energy))
        self.health = max(0.0, min(100.0, self.health))
        self.curiosity = max(0.0, min(100.0, self.curiosity))

        self.visited.add((self.x, self.y))

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
            self.known_cells[
                (observation.x, observation.y)
            ] = observation.cell

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
        self.visited.add((x, y))

    def record_experience(
        self,
        outcome: ActionOutcome,
    ) -> None:
        self.experiences.append(
            Experience(
                position=outcome.position,
                event=outcome.event,
                energy_change=outcome.energy_change,
                health_change=outcome.health_change,
                curiosity_change=outcome.curiosity_change,
            )
        )

    def is_alive(self) -> bool:
        return self.energy > 0 and self.health > 0
