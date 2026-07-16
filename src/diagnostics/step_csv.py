import csv
from pathlib import Path
from typing import Any, Sequence

from src.diagnostics.csv_values import enum_csv, optional_csv
from src.diagnostics.outcome_csv import outcome_metrics_to_csv_row
from src.diagnostics.step_csv_schema import STEP_CSV_FIELDNAMES
from src.simulation.step_action_text import format_action
from src.telemetry.metrics import Position, StepMetrics


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
    goal_target_x, goal_target_y = position_to_csv(metrics.goal_target)
    food_x, food_y = position_to_csv(metrics.memory_last_food_position)
    mystery_x, mystery_y = position_to_csv(
        metrics.memory_last_mystery_position
    )
    danger_x, danger_y = position_to_csv(
        metrics.memory_last_danger_position
    )

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
        "memory_goal_age": metrics.memory_goal_age,
        "memory_goal_switch_count": metrics.memory_goal_switch_count,
        "memory_recent_revisit_count": (
            metrics.memory_recent_revisit_count
        ),
        "memory_stuck_counter": metrics.memory_stuck_counter,
        "memory_recent_rest_count": metrics.memory_recent_rest_count,
        "memory_energy_trend": metrics.memory_energy_trend,
        "memory_last_food_x": food_x,
        "memory_last_food_y": food_y,
        "memory_last_mystery_x": mystery_x,
        "memory_last_mystery_y": mystery_y,
        "memory_last_danger_x": danger_x,
        "memory_last_danger_y": danger_y,
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


def position_to_csv(
    position: Position | None,
) -> tuple[int | str, int | str]:
    if position is None:
        return "", ""

    return position
