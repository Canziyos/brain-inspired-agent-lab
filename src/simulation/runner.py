from pathlib import Path

from src.configs import SimulationConfig
from src.diagnostics.run_setup import (
    configure_run_logging,
    create_run_directory,
)
from src.memory.episode_store import save_episode_store
from src.simulation.loop import run_simulation_loop
from src.simulation.reporting import (
    log_startup,
    log_termination,
    show_run_views,
    write_run_outputs,
)
from src.simulation.session import create_simulation_session
from src.telemetry.metrics import StepMetrics



def run_simulation(
    config: SimulationConfig,
) -> list[StepMetrics]:
    run_directory = (
        create_run_directory(config.output.output_root)
        if config.output.save_run_outputs
        else None
    )

    logger = configure_run_logging(
        run_directory=run_directory,
        debug=config.runtime.verbose,
    )

    session = create_simulation_session(config)

    log_startup(
        logger=logger,
        config=config,
        run_directory=run_directory,
        prior_episode_count=len(session.prior_episodes),
    )

    try:
        result = run_simulation_loop(
            config=config,
            session=session,
        )

        log_termination(
            logger=logger,
            history=result.history,
        )

        write_run_outputs(
            logger=logger,
            history=result.history,
            run_directory=run_directory,
            terminated=result.terminated,
            truncated=result.truncated,
            episodes=session.episodic_trace.episodes,
            working_memory=session.working_memory,
            world_width=config.world.width,
            world_height=config.world.height,
        )

        saved_memory_path = save_episode_store(
            config=config,
            episodes=session.episodic_trace.episodes,
            source_run_id=run_identifier(
                run_directory=run_directory,
                config=config,
            ),
        )

        if saved_memory_path is not None:
            logger.info(
                "Persistent episodic memory written to: %s",
                saved_memory_path,
            )

        show_run_views(
            config=config,
            env=session.env,
            history=result.history,
        )

        return result.history

    finally:
        session.env.close()



def run_identifier(
    run_directory: Path | None,
    config: SimulationConfig,
) -> str:
    if run_directory is not None:
        return run_directory.name

    return f"seed-{config.runtime.random_seed}"
