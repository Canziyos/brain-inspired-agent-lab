import csv

from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.core.world import CellType
from src.diagnostics.step_csv import write_steps_csv
from src.telemetry.metrics import (
    OutcomeModelMetrics,
    StepMetrics,
)


def test_write_steps_csv_writes_step_metrics(tmp_path) -> None:
    output_path = tmp_path / "steps.csv"

    history = [
        StepMetrics(
            step=3,
            position=(1, 2),
            energy=91.0,
            health=100.0,
            curiosity=45.0,
            reward=6.5,
            predicted_reward=5.25,
            loss=None,
            event=EventType.DISCOVERED_MYSTERY,
            visited_count=4,
            known_cell_count=9,
            choices_agree=True,
            grid_snapshot=((CellType.EMPTY, CellType.MYSTERY),),
            rule_action=Action.MOVE_EAST,
            action_reason="curious about mystery",
            network_action=Action.REST,
            imagination_action=Action.MOVE_EAST,
            imagination_expected_reward=7.25,
            imagination_utility=18.625,
            rule_imagined_reward=7.25,
            rule_imagined_utility=18.625,
            imagination_agrees=True,
            termination_reason=None,
            goal_kind="mystery",
            goal_target=(1, 2),
            outcome_model=OutcomeModelMetrics(
                predicted_energy_change=-1.0,
                actual_energy_change=-2.0,
                predicted_health_change=0.0,
                actual_health_change=0.0,
                predicted_curiosity_change=4.0,
                actual_curiosity_change=5.0,
                predicted_event=EventType.ATE_FOOD,
                actual_event=EventType.DISCOVERED_MYSTERY,
                event_correct=False,
                state_mae=1.0,
                total_loss=None,
                state_loss=None,
                event_loss=None,
                neural_state_mean=0.15,
                neural_state_std=0.05,
                neural_state_min=0.1,
                neural_state_max=0.2,
                reset_state_mae=1.5,
                persistent_better_than_reset=True,
            ),
        )
    ]

    write_steps_csv(history, output_path)

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "step": "3",
            "position_x": "1",
            "position_y": "2",
            "energy": "91.0",
            "health": "100.0",
            "curiosity": "45.0",
            "reward": "6.5",
            "predicted_reward": "5.25",
            "loss": "",
            "event": "discovered_mystery",
            "visited_count": "4",
            "known_cell_count": "9",
            "goal_kind": "mystery",
            "goal_target_x": "1",
            "goal_target_y": "2",
            "rule_action": "move move_east (+1, +0)",
            "action_reason": "curious about mystery",
            "network_action": "rest",
            "choices_agree": "True",
            "imagination_action": "move move_east (+1, +0)",
            "imagination_expected_reward": "7.25",
            "imagination_utility": "18.625",
            "rule_imagined_reward": "7.25",
            "rule_imagined_utility": "18.625",
            "imagination_agrees": "True",
            "termination_reason": "",
            "outcome_predicted_energy_change": "-1.0",
            "outcome_actual_energy_change": "-2.0",
            "outcome_predicted_health_change": "0.0",
            "outcome_actual_health_change": "0.0",
            "outcome_predicted_curiosity_change": "4.0",
            "outcome_actual_curiosity_change": "5.0",
            "outcome_predicted_event": "ate_food",
            "outcome_actual_event": "discovered_mystery",
            "outcome_event_correct": "False",
            "outcome_state_mae": "1.0",
            "outcome_reset_state_mae": "1.5",
            "outcome_persistent_better_than_reset": "True",
            "outcome_total_loss": "",
            "outcome_state_loss": "",
            "outcome_event_loss": "",
            "outcome_neural_state_mean": "0.15",
            "outcome_neural_state_std": "0.05",
            "outcome_neural_state_min": "0.1",
            "outcome_neural_state_max": "0.2",
        }
    ]
