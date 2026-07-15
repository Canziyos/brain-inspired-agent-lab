from collections import deque
from collections.abc import Collection
from typing import TypeAlias

from src.core.actions import MOVE_DELTAS
from src.core.agent import Agent
from src.core.world import CellType


Position: TypeAlias = tuple[int, int]


def neighbor_positions(
    position: Position,
    width: int,
    height: int,
) -> tuple[Position, ...]:
    if width <= 0 or height <= 0:
        raise ValueError("World width and height must be positive")

    x, y = position
    result: list[Position] = []

    for dx, dy in MOVE_DELTAS:
        adjacent = x + dx, y + dy
        ax, ay = adjacent

        if 0 <= ax < width and 0 <= ay < height:
            result.append(adjacent)

    return tuple(result)


def known_positions(agent: Agent) -> set[Position]:
    return set(agent.known_cells) | set(agent.visited)


def known_traversable_cells(agent: Agent) -> set[Position]:
    positions = known_positions(agent)

    traversable = {
        position
        for position in positions
        if agent.known_cells.get(position) is not CellType.DANGER
    }

    traversable.add(agent.position)
    return traversable


def find_frontiers(
    agent: Agent,
    width: int,
    height: int,
) -> set[Position]:
    positions = known_positions(agent)
    traversable = known_traversable_cells(agent)

    return {
        position
        for position in traversable
        if any(
            adjacent not in positions
            for adjacent in neighbor_positions(
                position,
                width,
                height,
            )
        )
    }


def shortest_path(
    start: Position,
    goals: Collection[Position],
    traversable: set[Position],
    width: int,
    height: int,
) -> list[Position] | None:
    remaining_goals = set(goals) - {start}

    if not remaining_goals:
        return None

    queue: deque[Position] = deque([start])
    parents: dict[Position, Position | None] = {
        start: None,
    }

    while queue:
        current = queue.popleft()

        if current in remaining_goals:
            return reconstruct_path(
                parents=parents,
                goal=current,
            )

        for adjacent in neighbor_positions(
            current,
            width,
            height,
        ):
            if adjacent in parents:
                continue

            if adjacent not in traversable:
                continue

            parents[adjacent] = current
            queue.append(adjacent)

    return None


def reconstruct_path(
    parents: dict[Position, Position | None],
    goal: Position,
) -> list[Position]:
    path: list[Position] = []
    position: Position | None = goal

    while position is not None:
        path.append(position)
        position = parents[position]

    path.reverse()
    return path