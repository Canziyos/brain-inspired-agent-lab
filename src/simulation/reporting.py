import logging
from collections.abc import Sequence
from pathlib import Path

from src.configs import SimulationConfig
from src.diagnostics.coverage_csv import write_coverage_csv
from src.diagnostics.episode_csv import write_episodes_csv
from src.diagnostics.step_csv import write_steps_csv
from src.envs.grid_world_env import BabyViceGridEnv
from src.memory.episode_trace import Episode
from src.memory.working_memory import WorkingMemory
from src.telemetry.metrics import StepMetrics
from src.views.animation import animate_simulation
from src.views.plots import plot_simulation_summary



def log_startup(
    logger: logging.Logger,
    config: SimulationConfig,
    run_directory: Path | None,
) -> None:
    if run_directory is None:
        return

    logger.info(
        (
            "Simulation started: "
            "world=%dx%d, max_steps=%d, seed=%d"
        ),
        config.world.width,
        config.world.height,
        config.runtime.max_steps,
        config.runtime.random_seed,
    )

    logger.info(
        "Run directory: %s",
        run_directory,
    )



def log_termination(
    logger: logging.Logger,
    history: list[StepMetrics],
) -> None:
    if not history:
        return

    termination_reason = history[-1].termination_reason

    if termination_reason == "dead":
        logger.warning(
            "Baby Vice died after %d steps.",
            len(history),
        )

    elif termination_reason == "goals_complete":
        logger.info(
            (
                "Baby Vice completed all reachable goals "
                "after %d steps."
            ),
            len(history),
        )

    elif termination_reason == "time_limit":
        logger.info(
            "Simulation reached the time limit after %d steps.",
            len(history),
        )



def log_final_summary(
    logger: logging.Logger,
    history: list[StepMetrics],
    terminated: bool,
    truncated: bool,
) -> None:
    if not history:
        logger.warning(
            "Simulation produced no step metrics."
        )
        return

    final = history[-1]

    total_reward = sum(
        item.reward
        for item in history
    )

    mean_reward = total_reward / len(history)

    agreement_rate = (
        sum(item.choices_agree for item in history)
        / len(history)
    )

    imagination_agreement_rate = (
        sum(item.imagination_agrees for item in history)
        / len(history)
    )

    episodic_advice_count = sum(
        item.episodic_action is not None
        for item in history
    )
    episodic_advice_rate = episodic_advice_count / len(history)

    episodic_rule_agreement_rate = (
        sum(
            item.episodic_agrees_with_rule
            for item in history
        )
        / episodic_advice_count
        if episodic_advice_count > 0
        else 0.0
    )

    episodic_imagination_agreement_rate = (
        sum(
            item.episodic_agrees_with_imagination
            for item in history
        )
        / episodic_advice_count
        if episodic_advice_count > 0
        else 0.0
    )

    logger.info(
        (
            "Simulation finished: "
            "steps=%d, position=%s, "
            "visited=%d, known=%d, "
            "seen=%d/%d %.1f%%, "
            "unseen=%d, frontiers=%d, "
            "reachable_frontiers=%d, "
            "unreachable_frontiers=%d, "
            "frontier_clusters=%d, "
            "reachable_frontier_clusters=%d, "
            "energy=%.1f, health=%.1f, "
            "curiosity=%.1f, "
            "total_reward=%.2f, "
            "mean_reward=%.3f, "
            "reward_network_agreement=%.1f%%, "
            "imagination_agreement=%.1f%%, "
            "episodic_advice_rate=%.1f%%, "
            "episodic_rule_agreement=%.1f%%, "
            "episodic_imagination_agreement=%.1f%%, "
            "semantic_goal_switches=%d, "
            "target_switches=%d, "
            "frontier_target_switches=%d, "
            "frontier_semantic_switches=%d, "
            "stuck_counter=%d, "
            "terminated=%s, "
            "truncated=%s"
        ),
        len(history),
        final.position,
        final.visited_count,
        final.known_cell_count,
        final.coverage_seen_cell_count,
        final.coverage_total_world_cells,
        final.coverage_seen_ratio * 100.0,
        final.coverage_unseen_cell_count,
        final.coverage_frontier_count,
        final.coverage_reachable_frontier_count,
        final.coverage_unreachable_frontier_count,
        final.coverage_frontier_cluster_count,
        final.coverage_reachable_frontier_cluster_count,
        final.energy,
        final.health,
        final.curiosity,
        total_reward,
        mean_reward,
        agreement_rate * 100.0,
        imagination_agreement_rate * 100.0,
        episodic_advice_rate * 100.0,
        episodic_rule_agreement_rate * 100.0,
        episodic_imagination_agreement_rate * 100.0,
        final.memory_goal_switch_count,
        final.memory_target_switch_count,
        final.memory_frontier_target_switch_count,
        final.memory_frontier_semantic_switch_count,
        final.memory_stuck_counter,
        terminated,
        truncated,
    )



def write_run_outputs(
    logger: logging.Logger,
    history: list[StepMetrics],
    run_directory: Path | None,
    terminated: bool,
    truncated: bool,
    episodes: Sequence[Episode],
    working_memory: WorkingMemory,
    world_width: int,
    world_height: int,
) -> None:
    if run_directory is None:
        return

    steps_path = run_directory / "steps.csv"
    episodes_path = run_directory / "episodes.csv"
    coverage_path = run_directory / "coverage.csv"

    write_steps_csv(
        history=history,
        output_path=steps_path,
    )

    write_episodes_csv(
        episodes=episodes,
        output_path=episodes_path,
    )

    write_coverage_csv(
        memory=working_memory,
        width=world_width,
        height=world_height,
        output_path=coverage_path,
    )

    log_final_summary(
        logger=logger,
        history=history,
        terminated=terminated,
        truncated=truncated,
    )

    logger.info(
        "Step metrics written to: %s",
        steps_path,
    )
    logger.info(
        "Episode trace written to: %s",
        episodes_path,
    )
    logger.info(
        "Coverage trace written to: %s",
        coverage_path,
    )



def show_run_views(
    config: SimulationConfig,
    env: BabyViceGridEnv,
    history: list[StepMetrics],
) -> None:
    if config.output.show_animation:
        animate_simulation(
            history=history,
            interval_ms=config.output.animation_interval_ms,
        )

    if config.output.show_plots:
        assert env.world is not None

        plot_simulation_summary(
            world=env.world,
            history=history,
        )
