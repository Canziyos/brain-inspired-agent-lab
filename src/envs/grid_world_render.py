from typing import Any

import numpy as np

from src.core.agent import Agent
from src.core.world import CellType, World


CELL_COLORS = {
    CellType.EMPTY: (236, 236, 236),
    CellType.FOOD: (91, 174, 87),
    CellType.DANGER: (191, 72, 72),
    CellType.MYSTERY: (116, 105, 182),
}

BACKGROUND_COLOR = (32, 34, 37)
GRID_LINE_COLOR = (45, 48, 51)
AGENT_COLOR = (35, 87, 164)


class GridWorldRenderer:
    def __init__(
        self,
        render_mode: str | None,
        cell_size: int,
        render_fps: int,
    ) -> None:
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.render_fps = render_fps

        self._pygame: Any | None = None
        self._window: Any | None = None
        self._clock: Any | None = None

    def render(
        self,
        world: World,
        agent: Agent,
    ):
        pygame = self._load_pygame()

        width_px = world.width * self.cell_size
        height_px = world.height * self.cell_size

        if self.render_mode == "human":
            surface = self._human_surface(
                pygame=pygame,
                width_px=width_px,
                height_px=height_px,
            )
        else:
            surface = pygame.Surface(
                (width_px, height_px)
            )

        self._draw(
            surface=surface,
            world=world,
            agent=agent,
        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

            if self._clock is None:
                self._clock = pygame.time.Clock()

            self._clock.tick(self.render_fps)
            return None

        return np.transpose(
            pygame.surfarray.array3d(surface),
            axes=(1, 0, 2),
        )

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()

        self._pygame = None
        self._window = None
        self._clock = None

    def _human_surface(
        self,
        pygame,
        width_px: int,
        height_px: int,
    ):
        if self._window is None:
            self._window = pygame.display.set_mode(
                (width_px, height_px)
            )
            pygame.display.set_caption(
                "Baby Vice GridWorld"
            )

        return self._window

    def _draw(
        self,
        surface,
        world: World,
        agent: Agent,
    ) -> None:
        pygame = self._load_pygame()

        surface.fill(BACKGROUND_COLOR)

        for y, row in enumerate(world.grid):
            for x, cell in enumerate(row):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                pygame.draw.rect(
                    surface,
                    CELL_COLORS.get(
                        cell,
                        CELL_COLORS[CellType.EMPTY],
                    ),
                    rect,
                )

                pygame.draw.rect(
                    surface,
                    GRID_LINE_COLOR,
                    rect,
                    1,
                )

        center = (
            agent.x * self.cell_size
            + self.cell_size // 2,
            agent.y * self.cell_size
            + self.cell_size // 2,
        )

        pygame.draw.circle(
            surface,
            AGENT_COLOR,
            center,
            self.cell_size // 3,
        )

    def _load_pygame(self):
        if self._pygame is None:
            import pygame

            pygame.init()
            self._pygame = pygame

        return self._pygame