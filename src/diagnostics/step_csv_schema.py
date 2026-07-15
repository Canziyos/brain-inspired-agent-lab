from typing import Final


STEP_FIELDS: Final[tuple[str, ...]] = (
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
    "imagination_action",
    "imagination_expected_reward",
    "imagination_utility",
    "rule_imagined_reward",
    "rule_imagined_utility",
    "imagination_agrees",
    "termination_reason",
)

OUTCOME_FIELDS: Final[tuple[str, ...]] = (
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
    "outcome_reset_state_mae",
    "outcome_persistent_better_than_reset",
    "outcome_total_loss",
    "outcome_state_loss",
    "outcome_event_loss",
    "outcome_neural_state_mean",
    "outcome_neural_state_std",
    "outcome_neural_state_min",
    "outcome_neural_state_max",
)

STEP_CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    STEP_FIELDS + OUTCOME_FIELDS
)