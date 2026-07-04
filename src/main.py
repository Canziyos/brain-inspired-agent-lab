from src.config import SimulationConfig
from src.simulation.runner import run_simulation


def main() -> None:
    config = SimulationConfig(
        max_steps=500,
        verbose=True,
        show_animation=True,
        show_plots=False,
        animation_interval_ms=200,
    )

    run_simulation(config)


if __name__ == "__main__":
    main()
