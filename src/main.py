from dataclasses import replace

from src.configs import SimulationConfig
from src.simulation.runner import run_simulation


def create_demo_config() -> SimulationConfig:
    base_config = SimulationConfig()

    return replace(
        base_config,
        output=replace(
            base_config.output,
            show_animation=True,
            show_plots=False,
            animation_interval_ms=200,
            save_run_outputs=True,
        ),
    )


def main() -> None:
    run_simulation(create_demo_config())


if __name__ == "__main__":
    main()