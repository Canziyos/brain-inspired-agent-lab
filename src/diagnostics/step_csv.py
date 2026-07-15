import csv
from pathlib import Path
from typing import Any, Sequence

from src.diagnostics.csv_values import enum_csv, optional_csv
from src.diagnostics.outcome_csv import outcome_metrics_to_csv_row
from src.diagnostics.step_csv_schema import STEP_CSV_FIELDNAMES
from src.simulation.step_action_text import format_action
from src.telemetry.metrics import StepMetrics


def write_steps_csv(
    history: Sequence[StepMetrics],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=STEP_CSV_FIELDNAMES,
        )

        writer.writeheader()

        for metrics in history:
            writer.writerow(step_metrics_to_csv_row(metrics))


def step_metrics_to_csv_row(metrics: StepMetrics) -> dict[str, Any]:
    goal_target_x: int | str = ""
    goal_target_y: int | str = ""

    if metrics.goal_target is not None:
        goal_target_x, goal_target_y = metrics.goal_target

    row: dict[str, Any] = {
        "step": metrics.step,
        "position_x": metrics.position[0],
        "position_y": metrics.position[1],
        "energy": metrics.energy,
        "health": metrics.health,
        "curiosity": metrics.curiosity,
        "reward": metrics.reward,
        "predicted_reward": metrics.predicted_reward,
        "loss": optional_csv(metrics.loss),
        "event": enum_csv(metrics.event),
        "visited_count": metrics.visited_count,
        "known_cell_count": metrics.known_cell_count,
        "goal_kind": optional_csv(metrics.goal_kind),
        "goal_target_x": goal_target_x,
        "goal_target_y": goal_target_y,
        "rule_action": format_action(metrics.rule_action),
        "action_reason": metrics.action_reason,
        "network_action": format_action(metrics.network_action),
        "choices_agree": metrics.choices_agree,
        "imagination_action": format_action(metrics.imagination_action),
        "imagination_expected_reward": (
            metrics.imagination_expected_reward
        ),
        "imagination_utility": metrics.imagination_utility,
        "rule_imagined_reward": metrics.rule_imagined_reward,
        "rule_imagined_utility": metrics.rule_imagined_utility,
        "imagination_agrees": metrics.imagination_agrees,
        "termination_reason": optional_csv(
            metrics.termination_reason
        ),
    }

    row.update(outcome_metrics_to_csv_row(metrics.outcome_model))

    return row
