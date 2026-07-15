import random

import pytest

from src.core.world import CellType, World


def test_place_random_reserves_start_cell() -> None:
    world = World(width=2, height=1)

    world.place_random(
        CellType.FOOD,
        1,
        rng=random.Random(0),
        reserved={(0, 0)},
    )

    assert world.get_cell(0, 0) is CellType.EMPTY
    assert world.get_cell(1, 0) is CellType.FOOD


def test_place_random_rejects_over_capacity_requests() -> None:
    world = World(width=1, height=1)

    with pytest.raises(ValueError):
        world.place_random(
            CellType.FOOD,
            1,
            reserved={(0, 0)},
        )
