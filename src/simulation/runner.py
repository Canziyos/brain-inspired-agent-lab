import random

import torch

from src.config import SimulationConfig
from src.diagnostics.run_output import (
    configure_run_logging,
    create_run_directory,
    write_steps_csv,
)
from src.envs.grid_world_env import BabyViceGridEnv
from src.learning.samples import (
    OutcomeSample,
    RewardSample,
)
from src.simulation.metrics import StepMetrics
from src.simulation.setup import (
    create_outcome_model,
    create_reward_network,
)
from src.simulation.step import run_simulation_step
from src.views.animation import animate_simulation
from src.views.plots import plot_simulation_summary


def run_simulation(
    config: SimulationConfig,
) -> list[StepMetrics]:
    policy_rng = random.Random(
        config.random_seed + 1
    )

    reward_training_rng = random.Random(
        config.random_seed + 2
    )

    outcome_training_rng = random.Random(
        config.random_seed + 3
    )

    run_directory = None

    if config.save_run_outputs:
        run_directory = create_run_directory(
            config.output_root
        )

    logger = configure_run_logging(
        run_directory=run_directory,
        debug=config.verbose,
    )

    if run_directory is not None:
        logger.info(
            (
                "Simulation started: "
                "world=%dx%d, max_steps=%d, seed=%d"
            ),
            config.world_width,
            config.world_height,
            config.max_steps,
            config.random_seed,
        )

        logger.info(
            "Run directory: %s",
            run_directory,
        )

    env = BabyViceGridEnv(config=config)
    env.reset(seed=config.random_seed)

    # Preserve the original reward-network
    # initialisation sequence.
    torch.manual_seed(config.torch_seed)

    (
        reward_network,
        reward_optimizer,
    ) = create_reward_network(config)

    # Initialise the recurrent outcome model with
    # a separate deterministic seed.
    torch.manual_seed(
        config.torch_seed + 1
    )

    (
        outcome_model,
        outcome_optimizer,
    ) = create_outcome_model(config)

    reward_samples: list[RewardSample] = []
    outcome_samples: list[OutcomeSample] = []

    history: list[StepMetrics] = []

    agreement_count = 0
    comparison_count = 0

    terminated = False
    truncated = False
    step = 0

    try:
        while not terminated and not truncated:
            (
                metrics,
                agreement_count,
                comparison_count,
                terminated,
                truncated,
            ) = run_simulation_step(
                step=step,
                config=config,
                env=env,

                reward_network=reward_network,
                reward_optimizer=reward_optimizer,
                reward_samples=reward_samples,

                outcome_model=outcome_model,
                outcome_optimizer=outcome_optimizer,
                outcome_samples=outcome_samples,

                policy_rng=policy_rng,
                training_rng=reward_training_rng,
                outcome_training_rng=(
                    outcome_training_rng
                ),

                agreement_count=agreement_count,
                comparison_count=comparison_count,
            )

            history.append(metrics)
            step += 1

        if history:
            termination_reason = (
                history[-1].termination_reason
            )

            if termination_reason == "dead":
                logger.warning(
                    "Baby Vice died after %d steps.",
                    step,
                )

            elif termination_reason == (
                "goals_complete"
            ):
                logger.info(
                    (
                        "Baby Vice completed all "
                        "reachable goals after "
                        "%d steps."
                    ),
                    step,
                )

            elif termination_reason == (
                "time_limit"
            ):
                logger.info(
                    (
                        "Simulation reached the "
                        "time limit after %d steps."
                    ),
                    step,
                )

        assert env.world is not None

        if run_directory is not None:
            steps_path = (
                run_directory / "steps.csv"
            )

            write_steps_csv(
                history=history,
                output_path=steps_path,
            )

            if history:
                final = history[-1]

                total_reward = sum(
                    item.reward
                    for item in history
                )

                mean_reward = (
                    total_reward / len(history)
                )

                agreement_rate = (
                    agreement_count
                    / comparison_count
                    if comparison_count > 0
                    else 0.0
                )

                logger.info(
                    (
                        "Simulation finished: "
                        "steps=%d, position=%s, "
                        "visited=%d, known=%d, "
                        "energy=%.1f, health=%.1f, "
                        "curiosity=%.1f, "
                        "total_reward=%.2f, "
                        "mean_reward=%.3f, "
                        "agreement=%.1f%%, "
                        "terminated=%s, "
                        "truncated=%s"
                    ),
                    len(history),
                    final.position,
                    final.visited_count,
                    final.known_cell_count,
                    final.energy,
                    final.health,
                    final.curiosity,
                    total_reward,
                    mean_reward,
                    agreement_rate * 100.0,
                    terminated,
                    truncated,
                )

            else:
                logger.warning(
                    "Simulation produced no step metrics."
                )

            logger.info(
                "Step metrics written to: %s",
                steps_path,
            )

        if config.show_animation:
            animate_simulation(
                history=history,
                interval_ms=(
                    config.animation_interval_ms
                ),
            )

        if config.show_plots:
            plot_simulation_summary(
                world=env.world,
                history=history,
            )

        return history

    finally:
        env.close()