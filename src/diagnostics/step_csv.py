import csv
from pathlib import Path
from typing import Any, Sequence

from src.core.actions import Action
from src.core.dynamics_types import EventType
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
        "memory_goal_id": optional_csv(metrics.memory_goal_id),
        "memory_goal_age": metrics.memory_goal_age,
        "memory_goal_switch_count": metrics.memory_goal_switch_count,
        "memory_target_switch_count": metrics.memory_target_switch_count,
        "memory_frontier_target_switch_count": (
            metrics.memory_frontier_target_switch_count
        ),
        "memory_frontier_semantic_switch_count": (
            metrics.memory_frontier_semantic_switch_count
        ),
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
        "coverage_total_world_cells": (
            metrics.coverage_total_world_cells
        ),
        "coverage_seen_cell_count": (
            metrics.coverage_seen_cell_count
        ),
        "coverage_visited_cell_count": (
            metrics.coverage_visited_cell_count
        ),
        "coverage_unseen_cell_count": (
            metrics.coverage_unseen_cell_count
        ),
        "coverage_seen_ratio": metrics.coverage_seen_ratio,
        "coverage_visited_ratio": metrics.coverage_visited_ratio,
        "coverage_frontier_count": metrics.coverage_frontier_count,
        "coverage_reachable_frontier_count": (
            metrics.coverage_reachable_frontier_count
        ),
        "coverage_unreachable_frontier_count": (
            metrics.coverage_unreachable_frontier_count
        ),
        "coverage_frontier_cluster_count": (
            metrics.coverage_frontier_cluster_count
        ),
        "coverage_reachable_frontier_cluster_count": (
            metrics.coverage_reachable_frontier_cluster_count
        ),
        "coverage_current_frontier_cluster_id": optional_csv(
            metrics.coverage_current_frontier_cluster_id
        ),
        "coverage_newly_seen_count": (
            metrics.coverage_newly_seen_count
        ),
        "coverage_newly_visited_count": (
            metrics.coverage_newly_visited_count
        ),
        "coverage_first_full_seen_step": optional_csv(
            metrics.coverage_first_full_seen_step
        ),
        "coverage_first_full_visited_step": optional_csv(
            metrics.coverage_first_full_visited_step
        ),
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
        "episodic_action": optional_action_csv(metrics.episodic_action),
        "episodic_expected_reward": metrics.episodic_expected_reward,
        "episodic_confidence": metrics.episodic_confidence,
        "episodic_match_count": metrics.episodic_match_count,
        "episodic_best_event": optional_event_csv(
            metrics.episodic_best_event
        ),
        "episodic_risk_hit_danger": metrics.episodic_risk_hit_danger,
        "episodic_agrees_with_rule": metrics.episodic_agrees_with_rule,
        "episodic_agrees_with_imagination": (
            metrics.episodic_agrees_with_imagination
        ),
        "episodic_rationale": metrics.episodic_rationale,
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



def optional_action_csv(action: Action | None) -> str:
    if action is None:
        return ""

    return format_action(action)



def optional_event_csv(event: EventType | None) -> str:
    if event is None:
        return ""

    return enum_csv(event)
