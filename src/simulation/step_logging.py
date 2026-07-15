import logging

from src.simulation.step_action_text import format_action
from src.telemetry.metrics import (
    OutcomeModelMetrics,
    StepMetrics,
)


def log_step_metrics(
    logger: logging.Logger,
    metrics: StepMetrics,
) -> None:
    logger.debug(
        (
            "step=%d position=%s event=%s "
            "goal_kind=%s goal_target=%s "
            "rule_action=%s reason=%s "
            "network_action=%s "
            "imagination_action=%s "
            "imagination_expected_reward=%.2f "
            "imagination_utility=%.2f "
            "rule_imagined_reward=%.2f "
            "rule_imagined_utility=%.2f "
            "reward=%.2f predicted_reward=%.2f "
            "reward_loss=%s "
            "visited=%d known=%d agree=%s "
            "termination_reason=%s"
        ),
        metrics.step,
        metrics.position,
        metrics.event,
        metrics.goal_kind,
        metrics.goal_target,
        format_action(metrics.rule_action),
        metrics.action_reason,
        format_action(metrics.network_action),
        format_action(metrics.imagination_action),
        metrics.imagination_expected_reward,
        metrics.imagination_utility,
        metrics.rule_imagined_reward,
        metrics.rule_imagined_utility,
        metrics.reward,
        metrics.predicted_reward,
        metrics.loss,
        metrics.visited_count,
        metrics.known_cell_count,
        metrics.choices_agree,
        metrics.termination_reason,
    )


def log_outcome_metrics(
    logger: logging.Logger,
    metrics: OutcomeModelMetrics,
) -> None:
    logger.debug(
        (
            "outcome_model: "
            "delta_pred=(%.2f, %.2f, %.2f) "
            "delta_actual=(%.2f, %.2f, %.2f) "
            "event_pred=%s event_actual=%s "
            "event_correct=%s "
            "state_mae=%.3f "
            "reset_state_mae=%.3f "
            "persistent_better_than_reset=%s "
            "total_loss=%s "
            "state_loss=%s "
            "event_loss=%s"
        ),
        metrics.predicted_energy_change,
        metrics.predicted_health_change,
        metrics.predicted_curiosity_change,

        metrics.actual_energy_change,
        metrics.actual_health_change,
        metrics.actual_curiosity_change,

        metrics.predicted_event.name,
        metrics.actual_event.name,
        metrics.event_correct,
        metrics.state_mae,
        metrics.reset_state_mae,
        metrics.persistent_better_than_reset,

        metrics.total_loss,
        metrics.state_loss,
        metrics.event_loss,
    )
