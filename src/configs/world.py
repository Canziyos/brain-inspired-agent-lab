from dataclasses import dataclass

from src.configs.validation import (
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class WorldConfig:
    width: int = 18
    height: int = 16

    food_count: int = 5
    danger_count: int = 8
    mystery_count: int = 10

    def __post_init__(self) -> None:
        require_positive("world.width", self.width)
        require_positive("world.height", self.height)

        require_non_negative("world.food_count", self.food_count)
        require_non_negative("world.danger_count", self.danger_count)
        require_non_negative("world.mystery_count", self.mystery_count)

        total_cells = self.width * self.height
        placed_cells = (
            self.food_count
            + self.danger_count
            + self.mystery_count
        )

        if placed_cells >= total_cells:
            raise ValueError(
                "food_count + danger_count + mystery_count "
                "must leave at least one empty cell"
            )