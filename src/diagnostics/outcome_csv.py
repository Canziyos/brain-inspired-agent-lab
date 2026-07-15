from typing import Any

from src.diagnostics.csv_values import enum_csv, optional_csv
from src.diagnostics.step_csv_schema import OUTCOME_FIELDS
from src.telemetry.metrics import OutcomeModelMetrics


def outcome_metrics_to_csv_row(
    outcome: OutcomeModelMetrics | None,
) -> dict[str, Any]:
    if outcome is None:
        return empty_outcome_metrics_row()

    return {
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
        "outcome_predicted_event": enum_csv(outcome.predicted_event),
        "outcome_actual_event": enum_csv(outcome.actual_event),
        "outcome_event_correct": outcome.event_correct,
        "outcome_state_mae": outcome.state_mae,
        "outcome_reset_state_mae": outcome.reset_state_mae,
        "outcome_persistent_better_than_reset": (
            outcome.persistent_better_than_reset
        ),
        "outcome_total_loss": optional_csv(outcome.total_loss),
        "outcome_state_loss": optional_csv(outcome.state_loss),
        "outcome_event_loss": optional_csv(outcome.event_loss),
        "outcome_neural_state_mean": outcome.neural_state_mean,
        "outcome_neural_state_std": outcome.neural_state_std,
        "outcome_neural_state_min": outcome.neural_state_min,
        "outcome_neural_state_max": outcome.neural_state_max,
    }


def empty_outcome_metrics_row() -> dict[str, str]:
    return {
        field: ""
        for field in OUTCOME_FIELDS
    }