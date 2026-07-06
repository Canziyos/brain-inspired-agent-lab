import csv
import logging
from datetime import datetime
from pathlib import Path

from src.simulation.metrics import StepMetrics


LOGGER_NAME = "brain_lab"


def create_run_directory(
    output_root: str,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )

    run_directory = (
        Path(output_root)
        / timestamp
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def configure_run_logging(
    run_directory: Path | None,
    debug: bool,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.DEBUG
        if debug
        else logging.INFO
    )
    console_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s: %(message)s"
        )
    )

    logger.addHandler(console_handler)

    if run_directory is None:
        return logger

    file_handler = logging.FileHandler(
        run_directory / "run.log",
        encoding="utf-8",
    )
    file_handler.setLevel(
        logging.DEBUG
        if debug
        else logging.INFO
    )
    file_handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(file_handler)

    return logger


def write_steps_csv(
    history: list[StepMetrics],
    output_path: Path,
) -> None:
    fieldnames = [
        "step",
        "position_x",
        "position_y",
        "energy",
        "health",
        "curiosity",
        "reward",
        "predicted_reward",
        "loss",
        "event",
        "visited_count",
        "known_cell_count",
        "goal_kind",
        "goal_target_x",
        "goal_target_y",
        "rule_action",
        "action_reason",
        "network_action",
        "choices_agree",
        "termination_reason",

        "outcome_predicted_energy_change",
        "outcome_actual_energy_change",
        "outcome_predicted_health_change",
        "outcome_actual_health_change",
        "outcome_predicted_curiosity_change",
        "outcome_actual_curiosity_change",
        "outcome_predicted_event",
        "outcome_actual_event",
        "outcome_event_correct",
        "outcome_state_mae",
        "outcome_total_loss",
        "outcome_state_loss",
        "outcome_event_loss",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for metrics in history:
            goal_target_x: int | str = ""
            goal_target_y: int | str = ""

            if metrics.goal_target is not None:
                (
                    goal_target_x,
                    goal_target_y,
                ) = metrics.goal_target

            outcome = metrics.outcome_model

            writer.writerow(
                {
                    "step": metrics.step,
                    "position_x": metrics.position[0],
                    "position_y": metrics.position[1],
                    "energy": metrics.energy,
                    "health": metrics.health,
                    "curiosity": metrics.curiosity,
                    "reward": metrics.reward,

                    "predicted_reward": (
                        metrics.predicted_reward
                    ),

                    "loss": (
                        metrics.loss
                        if metrics.loss is not None
                        else ""
                    ),

                    "event": metrics.event,

                    "visited_count": (
                        metrics.visited_count
                    ),

                    "known_cell_count": (
                        metrics.known_cell_count
                    ),

                    "goal_kind": (
                        metrics.goal_kind
                        or ""
                    ),

                    "goal_target_x": goal_target_x,
                    "goal_target_y": goal_target_y,

                    "rule_action": (
                        metrics.rule_action
                    ),

                    "action_reason": (
                        metrics.action_reason
                    ),

                    "network_action": (
                        metrics.network_action
                    ),

                    "choices_agree": (
                        metrics.choices_agree
                    ),

                    "termination_reason": (
                        metrics.termination_reason
                        or ""
                    ),

                    "outcome_predicted_energy_change": (
                        outcome.predicted_energy_change
                    ),

                    "outcome_actual_energy_change": (
                        outcome.actual_energy_change
                    ),

                    "outcome_predicted_health_change": (
                        outcome.predicted_health_change
                    ),

                    "outcome_actual_health_change": (
                        outcome.actual_health_change
                    ),

                    "outcome_predicted_curiosity_change": (
                        outcome.predicted_curiosity_change
                    ),

                    "outcome_actual_curiosity_change": (
                        outcome.actual_curiosity_change
                    ),

                    "outcome_predicted_event": (
                        outcome.predicted_event
                    ),

                    "outcome_actual_event": (
                        outcome.actual_event
                    ),

                    "outcome_event_correct": (
                        outcome.event_correct
                    ),

                    "outcome_state_mae": (
                        outcome.state_mae
                    ),

                    "outcome_total_loss": (
                        outcome.total_loss
                        if outcome.total_loss is not None
                        else ""
                    ),

                    "outcome_state_loss": (
                        outcome.state_loss
                        if outcome.state_loss is not None
                        else ""
                    ),

                    "outcome_event_loss": (
                        outcome.event_loss
                        if outcome.event_loss is not None
                        else ""
                    ),
                }
            )