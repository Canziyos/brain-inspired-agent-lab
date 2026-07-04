from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Observation:
    x: int
    y: int
    cell: str


@dataclass(frozen=True, slots=True)
class AgentState:
    x: int
    y: int
    energy: float
    health: float
    curiosity: float
    visited: frozenset[tuple[int, int]]
