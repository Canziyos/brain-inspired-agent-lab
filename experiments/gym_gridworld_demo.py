from src.config import SimulationConfig
from src.envs.grid_world_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_REST,
    ACTION_RIGHT,
    ACTION_UP,
    BabyViceGridEnv,
)


KEY_TO_ACTION_NAME = {
    "up": ACTION_UP,
    "right": ACTION_RIGHT,
    "down": ACTION_DOWN,
    "left": ACTION_LEFT,
    "space": ACTION_REST,
}


def main() -> None:
    env = BabyViceGridEnv(
        config=SimulationConfig(
            max_steps=200,
            verbose=False,
            show_animation=False,
            show_plots=False,
        ),
        render_mode="human",
    )

    import pygame

    env.reset(seed=7)
    terminated = False
    truncated = False
    clock = pygame.time.Clock()

    while not terminated and not truncated:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            elif event.type == pygame.KEYDOWN:
                action = KEY_TO_ACTION_NAME.get(
                    pygame.key.name(event.key)
                )

        if action is not None:
            _, _, terminated, truncated, _ = env.step(action)

        env.render()
        clock.tick(env.metadata["render_fps"])

    env.close()


if __name__ == "__main__":
    main()
