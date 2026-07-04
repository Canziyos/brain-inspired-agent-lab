# This planner knows:
# map dimensions;
# observed cells;
# visited cells;
# known dangers.

# not where unseen food, danger, or mystery objects are.
from collections import deque

from src.core.agent import Agent
from src.core.world import DANGER


Position = tuple[int, int]

CARDINAL_DELTAS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def neighbors(
    position: Position,
    width: int,
    height: int,
) -> list[Position]:
    x, y = position
    result: list[Position] = []

    for dx, dy in CARDINAL_DELTAS:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))

    return result


def known_traversable_cells(
    agent: Agent,
) -> set[Position]:
    known_positions = (
        set(agent.known_cells)
        | set(agent.visited)
    )

    traversable = {
        position
        for position in known_positions
        if agent.known_cells.get(position) != DANGER
    }

    # The agent must always be able to plan outward
    # from its current position.
    traversable.add((agent.x, agent.y))

    return traversable


def find_frontiers(
    agent: Agent,
    width: int,
    height: int,
) -> set[Position]:
    known_positions = (
        set(agent.known_cells)
        | set(agent.visited)
    )

    traversable = known_traversable_cells(agent)

    return {
        position
        for position in traversable
        if any(
            adjacent not in known_positions
            for adjacent in neighbors(
                position,
                width,
                height,
            )
        )
    }


def shortest_path(
    start: Position,
    goals: set[Position],
    traversable: set[Position],
    width: int,
    height: int,
) -> list[Position] | None:
    remaining_goals = goals - {start}

    if not remaining_goals:
        return None

    queue: deque[Position] = deque([start])

    parents: dict[
        Position,
        Position | None,
    ] = {
        start: None,
    }

    while queue:
        current = queue.popleft()

        if current in remaining_goals:
            path: list[Position] = []
            position: Position | None = current

            while position is not None:
                path.append(position)
                position = parents[position]

            path.reverse()
            return path

        for adjacent in neighbors(
            current,
            width,
            height,
        ):
            if adjacent not in traversable:
                continue

            if adjacent in parents:
                continue

            parents[adjacent] = current
            queue.append(adjacent)

    return None


def plan_frontier_step(
    agent: Agent,
    width: int,
    height: int,
) -> Position | None:
    start = (agent.x, agent.y)

    traversable = known_traversable_cells(agent)

    frontiers = find_frontiers(
        agent,
        width,
        height,
    )

    path = shortest_path(
        start=start,
        goals=frontiers,
        traversable=traversable,
        width=width,
        height=height,
    )

    if path is None or len(path) < 2:
        return None

    return path[1]