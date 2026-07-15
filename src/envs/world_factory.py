import random
from collections.abc import Collection

from src.configs.world import WorldConfig
from src.core.world import CellType, World


Position = tuple[int, int]

AGENT_START_POSITION: Position = (0, 0)


def create_world(
    config: WorldConfig,
    random_seed: int,
    reserved: Collection[Position] = (AGENT_START_POSITION,),
) -> World:
    reserved_positions = set(reserved)

    _validate_reserved_positions(
        config=config,
        reserved=reserved_positions,
    )

    _validate_world_capacity(
        config=config,
        reserved_count=len(reserved_positions),
    )

    world = World(
        width=config.width,
        height=config.height,
    )

    rng = random.Random(random_seed)

    world.place_random(
        CellType.FOOD,
        config.food_count,
        rng=rng,
        reserved=reserved_positions,
    )

    world.place_random(
        CellType.DANGER,
        config.danger_count,
        rng=rng,
        reserved=reserved_positions,
    )

    world.place_random(
        CellType.MYSTERY,
        config.mystery_count,
        rng=rng,
        reserved=reserved_positions,
    )

    return world


def _validate_reserved_positions(
    config: WorldConfig,
    reserved: set[Position],
) -> None:
    invalid_positions = {
        position
        for position in reserved
        if not _is_inside_world(
            position=position,
            width=config.width,
            height=config.height,
        )
    }

    if invalid_positions:
        raise ValueError(
            "Reserved positions must be inside the world: "
            f"{sorted(invalid_positions)}"
        )


def _validate_world_capacity(
    config: WorldConfig,
    reserved_count: int,
) -> None:
    available_cells = (
        config.width * config.height
        - reserved_count
    )

    requested_cells = (
        config.food_count
        + config.danger_count
        + config.mystery_count
    )

    if requested_cells > available_cells:
        raise ValueError(
            "World content counts exceed available cells: "
            f"requested={requested_cells}, "
            f"available={available_cells}"
        )


def _is_inside_world(
    position: Position,
    width: int,
    height: int,
) -> bool:
    x, y = position

    return (
        0 <= x < width
        and 0 <= y < height
    )