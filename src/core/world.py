import random
from dataclasses import dataclass

EMPTY = "."
FOOD = "F"
DANGER = "D"
MYSTERY = "?"


@dataclass(slots=True)
class World:
    width: int
    height: int
    grid: list[list[str]] | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive")

        if self.grid is None:
            self.grid = [
                [EMPTY for _ in range(self.width)]
                for _ in range(self.height)
            ]

    def place_random(
        self,
        cell_type: str,
        count: int,
        rng: random.Random | None = None,
        reserved: set[tuple[int, int]] | None = None,
    ) -> None:
        if self.grid is None:
            raise ValueError("Grid is not initialized")

        if count < 0:
            raise ValueError("Cannot place a negative number of cells")

        random_source = rng if rng is not None else random
        reserved_cells = reserved if reserved is not None else set()
        available_cells = [
            (x, y)
            for y, row in enumerate(self.grid)
            for x, cell in enumerate(row)
            if cell == EMPTY and (x, y) not in reserved_cells
        ]

        if count > len(available_cells):
            raise ValueError(
                "Cannot place more cells than available empty space"
            )

        placed = 0

        while placed < count:
            x = random_source.randrange(self.width)
            y = random_source.randrange(self.height)

            if (
                self.grid[y][x] == EMPTY
                and (x, y) not in reserved_cells
            ):
                self.grid[y][x] = cell_type
                placed += 1

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> str | None:
        if not self.is_inside(x, y):
            return None

        if self.grid is None:
            return None

        return self.grid[y][x]

    def set_cell(self, x: int, y: int, value: str) -> None:
        if not self.is_inside(x, y):
            raise ValueError("Cannot set cell outside the world")

        if self.grid is None:
            raise ValueError("Grid is not initialized")

        self.grid[y][x] = value

    def neighbors(self, x: int, y: int) -> list[tuple[int, int, str]]:
        result = []

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx = x + dx
            ny = y + dy

            cell = self.get_cell(nx, ny)

            if cell is not None:
                result.append((nx, ny, cell))

        return result

    def display(self, agent_position: tuple[int, int]|None=None) -> None:
        if self.grid is None:
            raise ValueError("Grid is not initialized")

        for y in range(self.height):
            row_sumbols = []
            for x in range(self.width):
                if agent_position == (x, y):
                    row_sumbols.append("A")
                else:
                    row_sumbols.append(self.grid[y][x])

            print(" ".join(row_sumbols))
