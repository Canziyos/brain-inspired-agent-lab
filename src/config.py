from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    world_width: int = 16
    world_height: int = 12

    food_count: int = 5
    danger_count: int = 8
    mystery_count: int = 10

    max_steps: int = 500

    random_seed: int = 7
    torch_seed: int = 7

    learning_rate: float = 0.01
    batch_size: int = 8

    verbose: bool = True

    show_animation: bool = True
    show_plots: bool = True

    animation_interval_ms: int = 250

    save_run_outputs: bool = False
    output_root: str = "outputs/runs"
    outcome_learning_rate: float = 0.003
    outcome_neuron_count: int = 24
    outcome_neural_ticks: int = 8
    outcome_leak: float = 0.35