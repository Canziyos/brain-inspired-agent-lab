from dataclasses import dataclass
from typing import TypeAlias

from src.core.world import CellType


Position: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Observation:
    x: int
    y: int
    cell: CellType


@dataclass(frozen=True, slots=True)
class AgentState:
    x: int
    y: int
    energy: float
    health: float
    curiosity: float
    visited: frozenset[Position]
