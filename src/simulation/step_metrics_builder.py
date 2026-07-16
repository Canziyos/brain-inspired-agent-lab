from src.core.agent import Agent
from src.memory.working_memory import WorkingMemorySnapshot
from src.simulation.step_decision import StepDecision
from src.simulation.step_execution import StepExecution
from src.simulation.step_learning import StepLearningResult
from src.telemetry.metrics import (
    OutcomeModelMetrics,
    StepMetrics,
)



def build_step_metrics(
    step: int,
    agent: Agent,
    decision: StepDecision,
    execution: StepExecution,
    learning: StepLearningResult,
    outcome_metrics: OutcomeModelMetrics,
    memory: WorkingMemorySnapshot,
) -> StepMetrics:
    goal_kind = (
        decision.plan.kind.value
        if decision.plan is not None
        else None
    )

    goal_target = (
        decision.plan.target
        if decision.plan is not None
        else None
    )

    chosen_action = decision.rule_choice.action
    episodic_action = decision.episodic_advice.action
    same_run_action = decision.same_run_episodic_advice.action
    prior_action = decision.prior_episodic_advice.action

    return StepMetrics(
        step=step,
        position=agent.position,

        energy=agent.energy,
        health=agent.health,
        curiosity=agent.curiosity,

        reward=execution.reward,
        predicted_reward=(
            decision.chosen_reward_prediction.predicted_reward
        ),
        loss=learning.reward_loss,

        event=execution.event,
        visited_count=len(agent.visited),
        known_cell_count=len(agent.known_cells),

        choices_agree=decision.choices_agree,

        grid_snapshot=execution.grid_snapshot,

        rule_action=chosen_action,
        action_reason=decision.rule_choice.rationale,

        network_action=decision.network_choice.action,

        imagination_action=decision.imagined_choice.action,
        imagination_expected_reward=(
            decision.imagined_choice.expected_reward
        ),
        imagination_utility=(
            decision.imagined_choice.utility
        ),
        rule_imagined_reward=(
            decision.chosen_imagination.expected_reward
        ),
        rule_imagined_utility=(
            decision.chosen_imagination.utility
        ),
        imagination_agrees=decision.imagination_agrees,

        episodic_action=episodic_action,
        episodic_raw_expected_reward=(
            decision.episodic_advice.raw_expected_reward
        ),
        episodic_expected_reward=(
            decision.episodic_advice.expected_reward
        ),
        episodic_confidence=decision.episodic_advice.confidence,
        episodic_reliability=decision.episodic_advice.reliability,
        episodic_match_count=decision.episodic_advice.match_count,
        episodic_best_event=decision.episodic_advice.best_event,
        episodic_risk_hit_danger=(
            decision.episodic_advice.risk_hit_danger
        ),
        episodic_has_advice=decision.episodic_advice.has_advice,
        episodic_is_usable=decision.episodic_advice.is_usable,
        episodic_agrees_with_rule=(
            episodic_action is not None
            and episodic_action is chosen_action
        ),
        episodic_agrees_with_imagination=(
            episodic_action is not None
            and episodic_action is decision.imagined_choice.action
        ),
        episodic_reliability_reason=(
            decision.episodic_advice.reliability_reason
        ),
        episodic_rationale=decision.episodic_advice.rationale,

        episodic_prior_episode_count=decision.prior_episode_count,
        episodic_same_run_episode_count=decision.same_run_episode_count,

        episodic_same_run_action=same_run_action,
        episodic_same_run_expected_reward=(
            decision.same_run_episodic_advice.expected_reward
        ),
        episodic_same_run_confidence=(
            decision.same_run_episodic_advice.confidence
        ),
        episodic_same_run_reliability=(
            decision.same_run_episodic_advice.reliability
        ),
        episodic_same_run_is_usable=(
            decision.same_run_episodic_advice.is_usable
        ),
        episodic_same_run_agrees_with_rule=(
            same_run_action is not None
            and same_run_action is chosen_action
        ),
        episodic_same_run_agrees_with_imagination=(
            same_run_action is not None
            and same_run_action is decision.imagined_choice.action
        ),

        episodic_prior_action=prior_action,
        episodic_prior_expected_reward=(
            decision.prior_episodic_advice.expected_reward
        ),
        episodic_prior_confidence=(
            decision.prior_episodic_advice.confidence
        ),
        episodic_prior_reliability=(
            decision.prior_episodic_advice.reliability
        ),
        episodic_prior_is_usable=(
            decision.prior_episodic_advice.is_usable
        ),
        episodic_prior_agrees_with_rule=(
            prior_action is not None
            and prior_action is chosen_action
        ),
        episodic_prior_agrees_with_imagination=(
            prior_action is not None
            and prior_action is decision.imagined_choice.action
        ),

        termination_reason=execution.termination_reason,
        goal_kind=goal_kind,
        goal_target=goal_target,

        memory_goal_id=memory.current_goal_id,
        memory_goal_age=memory.current_goal_age,
        memory_goal_switch_count=memory.goal_switch_count,
        memory_target_switch_count=memory.target_switch_count,
        memory_frontier_target_switch_count=(
            memory.frontier_target_switch_count
        ),
        memory_frontier_semantic_switch_count=(
            memory.frontier_semantic_switch_count
        ),
        memory_recent_revisit_count=(
            memory.recent_position_revisit_count
        ),
        memory_stuck_counter=memory.stuck_counter,
        memory_recent_rest_count=memory.recent_rest_count,
        memory_energy_trend=memory.energy_trend,
        memory_last_food_position=memory.last_food_position,
        memory_last_mystery_position=memory.last_mystery_position,
        memory_last_danger_position=memory.last_danger_position,

        coverage_total_world_cells=memory.total_world_cells,
        coverage_seen_cell_count=memory.seen_cell_count,
        coverage_visited_cell_count=memory.visited_cell_count,
        coverage_unseen_cell_count=memory.unseen_cell_count,
        coverage_seen_ratio=memory.seen_ratio,
        coverage_visited_ratio=memory.visited_ratio,
        coverage_frontier_count=memory.frontier_count,
        coverage_reachable_frontier_count=(
            memory.reachable_frontier_count
        ),
        coverage_unreachable_frontier_count=(
            memory.unreachable_frontier_count
        ),
        coverage_frontier_cluster_count=memory.frontier_cluster_count,
        coverage_reachable_frontier_cluster_count=(
            memory.reachable_frontier_cluster_count
        ),
        coverage_current_frontier_cluster_id=(
            memory.current_frontier_cluster_id
        ),
        coverage_newly_seen_count=memory.newly_seen_count,
        coverage_newly_visited_count=memory.newly_visited_count,
        coverage_first_full_seen_step=(
            memory.first_full_seen_step
        ),
        coverage_first_full_visited_step=(
            memory.first_full_visited_step
        ),

        outcome_model=outcome_metrics,
    )
