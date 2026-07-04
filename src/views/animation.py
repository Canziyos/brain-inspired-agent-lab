import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.core.world import DANGER, EMPTY, FOOD, MYSTERY
from src.simulation.metrics import StepMetrics


def animate_simulation(
    history: list[StepMetrics],
    interval_ms: int = 250,
) -> None:
    if not history:
        print("No simulation history to animate.")
        return

    height = len(history[0].grid_snapshot)
    width = len(history[0].grid_snapshot[0])

    figure, axis = plt.subplots(
        figsize=(10, 7),
    )

    first_grid = grid_to_numeric(
        history[0].grid_snapshot
    )

    image = axis.imshow(
        first_grid,
        origin="upper",
        vmin=0,
        vmax=3,
    )

    agent_marker, = axis.plot(
        [],
        [],
        marker="o",
        markersize=14,
        linestyle="None",
        label="Baby Vice",
    )

    path_line, = axis.plot(
        [],
        [],
        linewidth=1.5,
        alpha=0.7,
        label="Path",
    )

    info_text = axis.text(
        0.02,
        1.02,
        "",
        transform=axis.transAxes,
        va="bottom",
    )

    action_text = axis.text(
        0.02,
        -0.12,
        "",
        transform=axis.transAxes,
        va="top",
    )

    axis.set_title("Baby Vice live simulation")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")

    axis.set_xticks(range(width))
    axis.set_yticks(range(height))
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)

    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")

    cell_labels: list = []

    def update(frame_index: int):
        metrics = history[frame_index]

        numeric_grid = grid_to_numeric(
            metrics.grid_snapshot
        )

        image.set_data(numeric_grid)

        agent_x, agent_y = metrics.position

        agent_marker.set_data(
            [agent_x],
            [agent_y],
        )

        path_positions = [
            item.position
            for item in history[:frame_index + 1]
        ]

        path_x = [
            position[0]
            for position in path_positions
        ]

        path_y = [
            position[1]
            for position in path_positions
        ]

        path_line.set_data(
            path_x,
            path_y,
        )

        info_text.set_text(
            f"Step {metrics.step} | "
            f"Energy {metrics.energy:.1f} | "
            f"Health {metrics.health:.1f} | "
            f"Curiosity {metrics.curiosity:.1f} | "
            f"Reward {metrics.reward:.2f}"
        )

        agreement_label = (
            "agree"
            if metrics.choices_agree
            else "disagree"
        )

        action_text.set_text(
            f"Rule: {metrics.rule_action}\n"
            f"Network: {metrics.network_action}\n"
            f"Policies: {agreement_label}"
        )

        for label in cell_labels:
            label.remove()

        cell_labels.clear()

        for y, row in enumerate(
            metrics.grid_snapshot
        ):
            for x, cell in enumerate(row):
                if cell != EMPTY:
                    label = axis.text(
                        x,
                        y,
                        cell,
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

                    cell_labels.append(label)

        return (
            image,
            agent_marker,
            path_line,
            info_text,
            action_text,
            *cell_labels,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(history),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    # Keep a reference so Python does not garbage-collect it.
    figure._baby_vice_animation = animation

    figure.tight_layout()
    plt.show()


def grid_to_numeric(
    grid: tuple[tuple[str, ...], ...],
) -> list[list[int]]:
    values = {
        EMPTY: 0,
        FOOD: 1,
        DANGER: 2,
        MYSTERY: 3,
    }

    return [
        [
            values.get(cell, 0)
            for cell in row
        ]
        for row in grid
    ]
