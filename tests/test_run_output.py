import csv

from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.core.world import CellType
from src.diagnostics.step_csv import write_steps_csv
from src.diagnostics.step_csv_schema import STEP_CSV_FIELDNAMES
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
            episodic_action=Action.MOVE_EAST,
            episodic_raw_expected_reward=4.0,
            episodic_expected_reward=2.5,
            episodic_confidence=0.6,
            episodic_reliability=0.7,
            episodic_match_count=5,
            episodic_best_event=EventType.DISCOVERED_MYSTERY,
            episodic_risk_hit_danger=0.0,
            episodic_has_advice=True,
            episodic_is_usable=True,
            episodic_agrees_with_rule=True,
            episodic_agrees_with_imagination=True,
            episodic_reliability_reason="usable",
            episodic_rationale="episodic advisor test rationale",
            episodic_prior_episode_count=100,
            episodic_same_run_episode_count=3,
            episodic_same_run_action=Action.MOVE_EAST,
            episodic_same_run_is_usable=True,
            episodic_same_run_agrees_with_rule=True,
            episodic_same_run_agrees_with_imagination=True,
            episodic_prior_action=Action.MOVE_EAST,
            episodic_prior_is_usable=True,
            episodic_prior_agrees_with_rule=True,
            episodic_prior_agrees_with_imagination=True,
            termination_reason=None,
            goal_kind="mystery",
            goal_target=(1, 2),
            memory_goal_id="mystery:1:2",
            memory_goal_age=4,
            memory_goal_switch_count=2,
            memory_target_switch_count=3,
            memory_frontier_target_switch_count=1,
            memory_frontier_semantic_switch_count=1,
            memory_recent_revisit_count=0,
            memory_stuck_counter=0,
            memory_recent_rest_count=1,
            memory_energy_trend=-3.0,
            memory_last_food_position=(0, 1),
            memory_last_mystery_position=(1, 2),
            memory_last_danger_position=None,
            coverage_total_world_cells=288,
            coverage_seen_cell_count=24,
            coverage_visited_cell_count=8,
            coverage_unseen_cell_count=264,
            coverage_seen_ratio=24 / 288,
            coverage_visited_ratio=8 / 288,
            coverage_frontier_count=6,
            coverage_reachable_frontier_count=5,
            coverage_unreachable_frontier_count=1,
            coverage_frontier_cluster_count=3,
            coverage_reachable_frontier_cluster_count=2,
            coverage_current_frontier_cluster_id="frontier:0:0",
            coverage_newly_seen_count=2,
            coverage_newly_visited_count=1,
            coverage_first_full_seen_step=None,
            coverage_first_full_visited_step=None,
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

    assert len(rows) == 1
    row = rows[0]

    assert tuple(row) == STEP_CSV_FIELDNAMES

    assert row["step"] == "3"
    assert row["position_x"] == "1"
    assert row["position_y"] == "2"
    assert row["energy"] == "91.0"
    assert row["event"] == "discovered_mystery"
    assert row["goal_kind"] == "mystery"
    assert row["goal_target_x"] == "1"
    assert row["goal_target_y"] == "2"
    assert row["rule_action"] == "move move_east (+1, +0)"
    assert row["action_reason"] == "curious about mystery"
    assert row["network_action"] == "rest"
    assert row["choices_agree"] == "True"
    assert row["imagination_action"] == "move move_east (+1, +0)"
    assert row["imagination_agrees"] == "True"

    assert row["episodic_action"] == "move move_east (+1, +0)"
    assert row["episodic_raw_expected_reward"] == "4.0"
    assert row["episodic_expected_reward"] == "2.5"
    assert row["episodic_confidence"] == "0.6"
    assert row["episodic_reliability"] == "0.7"
    assert row["episodic_match_count"] == "5"
    assert row["episodic_best_event"] == "discovered_mystery"
    assert row["episodic_risk_hit_danger"] == "0.0"
    assert row["episodic_has_advice"] == "True"
    assert row["episodic_is_usable"] == "True"
    assert row["episodic_agrees_with_rule"] == "True"
    assert row["episodic_agrees_with_imagination"] == "True"
    assert row["episodic_reliability_reason"] == "usable"
    assert row["episodic_rationale"] == "episodic advisor test rationale"
    assert row["episodic_prior_episode_count"] == "100"
    assert row["episodic_same_run_episode_count"] == "3"
    assert row["episodic_same_run_action"] == "move move_east (+1, +0)"
    assert row["episodic_same_run_is_usable"] == "True"
    assert row["episodic_same_run_agrees_with_rule"] == "True"
    assert row["episodic_same_run_agrees_with_imagination"] == "True"
    assert row["episodic_prior_action"] == "move move_east (+1, +0)"
    assert row["episodic_prior_is_usable"] == "True"
    assert row["episodic_prior_agrees_with_rule"] == "True"
    assert row["episodic_prior_agrees_with_imagination"] == "True"

    assert row["memory_goal_id"] == "mystery:1:2"
    assert row["memory_goal_age"] == "4"
    assert row["memory_goal_switch_count"] == "2"
    assert row["memory_target_switch_count"] == "3"
    assert row["memory_last_food_x"] == "0"
    assert row["memory_last_food_y"] == "1"
    assert row["memory_last_mystery_x"] == "1"
    assert row["memory_last_mystery_y"] == "2"
    assert row["memory_last_danger_x"] == ""
    assert row["memory_last_danger_y"] == ""

    assert row["coverage_total_world_cells"] == "288"
    assert row["coverage_seen_cell_count"] == "24"
    assert row["coverage_visited_cell_count"] == "8"
    assert row["coverage_unseen_cell_count"] == "264"
    assert row["coverage_frontier_count"] == "6"
    assert row["coverage_reachable_frontier_count"] == "5"
    assert row["coverage_unreachable_frontier_count"] == "1"
    assert row["coverage_frontier_cluster_count"] == "3"
    assert row["coverage_reachable_frontier_cluster_count"] == "2"
    assert row["coverage_current_frontier_cluster_id"] == "frontier:0:0"
    assert row["coverage_first_full_seen_step"] == ""
    assert row["coverage_first_full_visited_step"] == ""

    assert row["termination_reason"] == ""
    assert row["outcome_predicted_event"] == "ate_food"
    assert row["outcome_actual_event"] == "discovered_mystery"
    assert row["outcome_event_correct"] == "False"
    assert row["outcome_state_mae"] == "1.0"
    assert row["outcome_reset_state_mae"] == "1.5"
    assert row["outcome_persistent_better_than_reset"] == "True"
