from src.core.agent import Agent
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

        termination_reason=execution.termination_reason,
        goal_kind=goal_kind,
        goal_target=goal_target,

        outcome_model=outcome_metrics,
    )
