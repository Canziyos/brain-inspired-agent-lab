from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from typing import TypeAlias

from src.core.actions import MOVE_DELTAS
from src.core.agent import Agent
from src.core.world import CellType


Position: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class FrontierCluster:
    id: str
    cells: frozenset[Position]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def anchor(self) -> Position:
        return min(self.cells)



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



def reachable_positions(
    start: Position,
    traversable: set[Position],
    width: int,
    height: int,
) -> set[Position]:
    reachable: set[Position] = set()
    queue: deque[Position] = deque([start])

    traversable = set(traversable)
    traversable.add(start)

    while queue:
        current = queue.popleft()

        if current in reachable:
            continue

        reachable.add(current)

        for adjacent in neighbor_positions(
            current,
            width,
            height,
        ):
            if adjacent in reachable:
                continue

            if adjacent not in traversable:
                continue

            queue.append(adjacent)

    return reachable



def find_reachable_frontiers(
    agent: Agent,
    width: int,
    height: int,
) -> set[Position]:
    traversable = known_traversable_cells(agent)
    reachable = reachable_positions(
        start=agent.position,
        traversable=traversable,
        width=width,
        height=height,
    )

    return find_frontiers(
        agent=agent,
        width=width,
        height=height,
    ) & reachable



def cluster_positions(
    positions: Collection[Position],
    width: int,
    height: int,
) -> tuple[frozenset[Position], ...]:
    remaining = set(positions)
    clusters: list[frozenset[Position]] = []

    while remaining:
        start = min(remaining)
        queue: deque[Position] = deque([start])
        cluster: set[Position] = set()

        while queue:
            current = queue.popleft()

            if current not in remaining:
                continue

            remaining.remove(current)
            cluster.add(current)

            for adjacent in neighbor_positions(
                current,
                width,
                height,
            ):
                if adjacent in remaining:
                    queue.append(adjacent)

        clusters.append(frozenset(cluster))

    return tuple(
        sorted(
            clusters,
            key=lambda cluster: min(cluster),
        )
    )



def frontier_cluster_id(cells: Collection[Position]) -> str:
    if not cells:
        raise ValueError("Cannot create an id for an empty frontier cluster")

    anchor_x, anchor_y = min(cells)
    return f"frontier:{anchor_x}:{anchor_y}"



def find_frontier_clusters(
    agent: Agent,
    width: int,
    height: int,
    *,
    reachable_only: bool = False,
) -> tuple[FrontierCluster, ...]:
    frontiers = (
        find_reachable_frontiers(
            agent=agent,
            width=width,
            height=height,
        )
        if reachable_only
        else find_frontiers(
            agent=agent,
            width=width,
            height=height,
        )
    )

    return tuple(
        FrontierCluster(
            id=frontier_cluster_id(cluster),
            cells=cluster,
        )
        for cluster in cluster_positions(
            positions=frontiers,
            width=width,
            height=height,
        )
    )



def frontier_cluster_for_position(
    position: Position,
    clusters: Collection[FrontierCluster],
) -> FrontierCluster | None:
    for cluster in clusters:
        if position in cluster.cells:
            return cluster

    return None



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
