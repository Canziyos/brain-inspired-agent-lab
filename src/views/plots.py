from statistics import mean

import matplotlib.pyplot as plt

from src.core.world import CellType, World
from src.telemetry.metrics import StepMetrics


def plot_simulation_summary(
    world: World,
    history: list[StepMetrics],
) -> None:
    if not history:
        print("No simulation history to visualize.")
        return

    steps = [item.step for item in history]
    energy = [item.energy for item in history]
    health = [item.health for item in history]
    curiosity = [item.curiosity for item in history]

    rewards = [item.reward for item in history]
    predicted_rewards = [
        item.predicted_reward
        for item in history
    ]

    loss_steps = [
        item.step
        for item in history
        if item.loss is not None
    ]

    losses = [
        item.loss
        for item in history
        if item.loss is not None
    ]

    rolling_agreement = calculate_rolling_agreement(
        history,
        window_size=25,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    plot_final_world(
        axis=axes[0][0],
        world=world,
        history=history,
    )

    axes[0][1].plot(
        steps,
        energy,
        label="Energy",
    )
    axes[0][1].plot(
        steps,
        health,
        label="Health",
    )
    axes[0][1].plot(
        steps,
        curiosity,
        label="Curiosity",
    )

    axes[0][1].set_title("Internal state")
    axes[0][1].set_xlabel("Step")
    axes[0][1].set_ylabel("State value")
    axes[0][1].set_ylim(0, 105)
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    axes[1][0].plot(
        steps,
        rewards,
        label="Actual reward",
    )
    axes[1][0].plot(
        steps,
        predicted_rewards,
        label="Predicted reward",
    )

    axes[1][0].set_title(
        "Predicted versus actual reward"
    )
    axes[1][0].set_xlabel("Step")
    axes[1][0].set_ylabel("Reward")
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    if losses:
        axes[1][1].plot(
            loss_steps,
            losses,
            label="Training loss",
        )

    axes[1][1].plot(
        steps,
        rolling_agreement,
        label="Rolling agreement",
    )

    axes[1][1].set_title(
        "Learning and policy agreement"
    )
    axes[1][1].set_xlabel("Step")
    axes[1][1].set_ylabel("Loss / agreement")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    figure.suptitle(
        "Baby Vice simulation summary",
        fontsize=16,
    )

    figure.tight_layout()
    plt.show()


def plot_final_world(
    axis,
    world: World,
    history: list[StepMetrics],
) -> None:
    if world.grid is None:
        raise ValueError("World grid is not initialized")

    numeric_grid = []

    symbol_values = {
        CellType.EMPTY: 0,
        CellType.FOOD: 1,
        CellType.DANGER: 2,
        CellType.MYSTERY: 3,
    }

    for row in world.grid:
        numeric_grid.append(
            [
                symbol_values.get(cell, 0)
                for cell in row
            ]
        )

    axis.imshow(
        numeric_grid,
        origin="upper",
    )

    path_x = [
        item.position[0]
        for item in history
    ]

    path_y = [
        item.position[1]
        for item in history
    ]

    axis.plot(
        path_x,
        path_y,
        linewidth=1.5,
        alpha=0.7,
    )

    final_x, final_y = history[-1].position

    axis.scatter(
        [final_x],
        [final_y],
        s=120,
        marker="o",
        label="Baby Vice",
    )

    for y, row in enumerate(world.grid):
        for x, cell in enumerate(row):
            if cell is not CellType.EMPTY:
                axis.text(
                    x,
                    y,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    axis.set_title("Final world and movement path")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")

    axis.set_xticks(range(world.width))
    axis.set_yticks(range(world.height))

    axis.grid(True, alpha=0.3)
    axis.legend()


def calculate_rolling_agreement(
    history: list[StepMetrics],
    window_size: int,
) -> list[float]:
    agreement_values: list[float] = []

    for index in range(len(history)):
        window_start = max(
            0,
            index - window_size + 1,
        )

        window = history[
            window_start:index + 1
        ]

        agreement_rate = mean(
            1.0 if item.choices_agree else 0.0
            for item in window
        )

        agreement_values.append(
            agreement_rate
        )

    return agreement_values
