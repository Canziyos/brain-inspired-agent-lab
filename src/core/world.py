import random
from dataclasses import dataclass
from enum import StrEnum

from src.core.actions import MOVE_DELTAS


Position = tuple[int, int]


class CellType(StrEnum):
    EMPTY = "."
    FOOD = "F"
    DANGER = "D"
    MYSTERY = "?"


PLACEABLE_CELLS = frozenset(
    {
        CellType.FOOD,
        CellType.DANGER,
        CellType.MYSTERY,
    }
)


@dataclass(slots=True)
class World:
    width: int
    height: int
    grid: list[list[CellType]] | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "World width and height must be positive"
            )

        if self.grid is None:
            self.grid = [
                [CellType.EMPTY for _ in range(self.width)]
                for _ in range(self.height)
            ]
        else:
            self._validate_grid()

    def place_random(
        self,
        cell_type: CellType,
        count: int,
        rng: random.Random | None = None,
        reserved: set[Position] | None = None,
    ) -> None:
        if cell_type not in PLACEABLE_CELLS:
            raise ValueError(
                f"Cell type cannot be placed randomly: {cell_type!r}"
            )

        if count < 0:
            raise ValueError(
                "Cannot place a negative number of cells"
            )

        grid = self._require_grid()
        random_source = rng if rng is not None else random
        reserved_cells = reserved if reserved is not None else set()

        available_cells = [
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell is CellType.EMPTY
            and (x, y) not in reserved_cells
        ]

        if count > len(available_cells):
            raise ValueError(
                "Cannot place more cells than available empty space"
            )

        for x, y in random_source.sample(
            available_cells,
            count,
        ):
            grid[y][x] = cell_type

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> CellType | None:
        if not self.is_inside(x, y):
            return None

        grid = self._require_grid()
        return grid[y][x]

    def set_cell(
        self,
        x: int,
        y: int,
        value: CellType,
    ) -> None:
        if not self.is_inside(x, y):
            raise ValueError(
                "Cannot set cell outside the world"
            )

        self._validate_cell(value)

        grid = self._require_grid()
        grid[y][x] = value

    def neighbors(
        self,
        x: int,
        y: int,
    ) -> list[tuple[int, int, CellType]]:
        result: list[tuple[int, int, CellType]] = []

        for dx, dy in MOVE_DELTAS:
            nx = x + dx
            ny = y + dy

            cell = self.get_cell(nx, ny)

            if cell is not None:
                result.append((nx, ny, cell))

        return result

    def _require_grid(self) -> list[list[CellType]]:
        if self.grid is None:
            raise RuntimeError("Grid is not initialized")

        return self.grid

    def _validate_grid(self) -> None:
        grid = self._require_grid()

        if len(grid) != self.height:
            raise ValueError(
                "Grid height does not match world height"
            )

        for row in grid:
            if len(row) != self.width:
                raise ValueError(
                    "Grid width does not match world width"
                )

            for cell in row:
                self._validate_cell(cell)

    @staticmethod
    def _validate_cell(cell: CellType) -> None:
        if not isinstance(cell, CellType):
            raise ValueError(f"Unknown cell type: {cell!r}")
