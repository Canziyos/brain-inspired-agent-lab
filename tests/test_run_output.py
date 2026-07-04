import csv

from src.diagnostics.run_output import write_steps_csv
from src.simulation.metrics import StepMetrics


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
            event="DISCOVERED_MYSTERY",
            visited_count=4,
            known_cell_count=9,
            choices_agree=True,
            grid_snapshot=((".", "?"),),
            rule_action="move to (1, 2)",
            action_reason="curious about mystery",
            network_action="rest",
            termination_reason=None,
            goal_kind="mystery",
            goal_target=(1, 2),
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
            "event": "DISCOVERED_MYSTERY",
            "visited_count": "4",
            "known_cell_count": "9",
            "goal_kind": "mystery",
            "goal_target_x": "1",
            "goal_target_y": "2",
            "rule_action": "move to (1, 2)",
            "action_reason": "curious about mystery",
            "network_action": "rest",
            "choices_agree": "True",
            "termination_reason": "",
        }
    ]
