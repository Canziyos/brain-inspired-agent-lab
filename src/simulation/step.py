import logging
import random

import torch

from src.config import SimulationConfig
from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.envs.grid_world_env import (
    BabyViceGridEnv,
    action_to_discrete,
)
from src.learning.outcome_features import (
    encode_outcome_features,
)
from src.learning.outcome_model import (
    RateRecurrentOutcomeModel,
    event_index,
    predict_outcome_with_state,
    train_outcome_model,
)
from src.learning.reward_network import (
    ImmediateRewardNetwork,
    train_reward_network,
)
from src.learning.samples import (
    OutcomeSample,
    RewardSample,
)
from src.planning.goal_planner import (
    select_goal_plan,
)
from src.policies.neural_shadow import (
    actions_match,
    choose_network_action,
    find_action_prediction,
    predict_actions,
)
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)
from src.simulation.metrics import (
    OutcomeModelMetrics,
    StepMetrics,
)


logger = logging.getLogger(
    "brain_lab.simulation.step"
)


def format_action(action: Action) -> str:
    if action.kind == "rest":
        return "rest"

    return (
        f"move to "
        f"({action.target_x}, {action.target_y})"
    )


def run_simulation_step(
    step: int,
    config: SimulationConfig,
    env: BabyViceGridEnv,

    reward_network: ImmediateRewardNetwork,
    reward_optimizer: torch.optim.Optimizer,
    reward_samples: list[RewardSample],

    outcome_model: RateRecurrentOutcomeModel,
    outcome_optimizer: torch.optim.Optimizer,
    outcome_samples: list[OutcomeSample],

    policy_rng: random.Random,
    training_rng: random.Random,
    outcome_training_rng: random.Random,

    agreement_count: int,
    comparison_count: int,
    outcome_neural_state: torch.Tensor,
) -> tuple[StepMetrics, int, int, bool, bool, torch.Tensor]:
    if env.agent is None or env.world is None:
        raise RuntimeError(
            "Environment must be reset before simulation."
        )

    agent = env.agent
    world = env.world

    observations = agent.sense(world)
    agent.observe(observations)

    plan = select_goal_plan(
        agent=agent,
        width=config.world_width,
        height=config.world_height,
    )

    evaluations = evaluate_actions(
        agent,
        observations,
        plan=plan,
    )

    rule_choice = choose_action(
        evaluations,
        rng=policy_rng,
    )

    reward_predictions = predict_actions(
        agent=agent,
        evaluations=evaluations,
        model=reward_network,
    )

    network_choice = choose_network_action(
        reward_predictions
    )

    chosen_reward_prediction = (
        find_action_prediction(
            chosen_action=rule_choice.action,
            predictions=reward_predictions,
        )
    )

    choices_agree = actions_match(
        rule_choice.action,
        network_choice.action,
    )

    comparison_count += 1

    if choices_agree:
        agreement_count += 1

    chosen_action = rule_choice.action

    pre_action_state = agent.snapshot()

    perceived_cell = agent.known_cells.get(
        (
            chosen_action.target_x,
            chosen_action.target_y,
        )
    )

    outcome_features = encode_outcome_features(
        agent_state=pre_action_state,
        action=chosen_action,
        perceived_cell=perceived_cell,
    )

    state_before_prediction = outcome_neural_state.detach()

    outcome_prediction, updated_outcome_neural_state = (
        predict_outcome_with_state(
            model=outcome_model,
            features=outcome_features,
            neural_state=state_before_prediction,
        )
    )

    reset_neural_state = outcome_model.initial_neural_state(
        batch_size=1,
        device=state_before_prediction.device,
        dtype=state_before_prediction.dtype,
    )

    reset_outcome_prediction, _ = predict_outcome_with_state(
        model=outcome_model,
        features=outcome_features,
        neural_state=reset_neural_state,
    )

    discrete_action = action_to_discrete(
        position=(agent.x, agent.y),
        action=chosen_action,
    )

    (
        _observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(discrete_action)

    actual_event = EventType[
        info["event"]
    ]

    actual_state_changes = (
        float(info["energy_change"]),
        float(info["health_change"]),
        float(info["curiosity_change"]),
    )

    reward_samples.append(
        RewardSample(
            features=(
                chosen_reward_prediction.features
            ),
            reward=reward,
        )
    )

    reward_loss = train_reward_network(
        model=reward_network,
        optimizer=reward_optimizer,
        reward_samples=reward_samples,
        batch_size=config.batch_size,
        rng=training_rng,
    )

    outcome_samples.append(
        OutcomeSample(
            features=outcome_features,
            state_changes=actual_state_changes,
            event_index=event_index(
                actual_event
            ),
            neural_state=tuple(
                float(value)
                for value in state_before_prediction[0].tolist()
            ),
        )
    )

    outcome_training_result = (
        train_outcome_model(
            model=outcome_model,
            optimizer=outcome_optimizer,
            samples=outcome_samples,
            batch_size=config.batch_size,
            rng=outcome_training_rng,
        )
    )

    outcome_state_mae = (
        abs(
            outcome_prediction.energy_change
            - actual_state_changes[0]
        )
        + abs(
            outcome_prediction.health_change
            - actual_state_changes[1]
        )
        + abs(
            outcome_prediction.curiosity_change
            - actual_state_changes[2]
        )
    ) / 3.0

    reset_outcome_state_mae = (
        abs(
            reset_outcome_prediction.energy_change
            - actual_state_changes[0]
        )
        + abs(
            reset_outcome_prediction.health_change
            - actual_state_changes[1]
        )
        + abs(
            reset_outcome_prediction.curiosity_change
            - actual_state_changes[2]
        )
    ) / 3.0

    outcome_metrics = OutcomeModelMetrics(
        predicted_energy_change=(
            outcome_prediction.energy_change
        ),
        actual_energy_change=(
            actual_state_changes[0]
        ),

        predicted_health_change=(
            outcome_prediction.health_change
        ),
        actual_health_change=(
            actual_state_changes[1]
        ),

        predicted_curiosity_change=(
            outcome_prediction.curiosity_change
        ),
        actual_curiosity_change=(
            actual_state_changes[2]
        ),

        predicted_event=(
            outcome_prediction.event.name
        ),
        actual_event=actual_event.name,
        event_correct=(
            outcome_prediction.event
            == actual_event
        ),

        state_mae=outcome_state_mae,
        reset_state_mae=reset_outcome_state_mae,
        persistent_better_than_reset=(
            outcome_state_mae
            < reset_outcome_state_mae
        ),

        total_loss=(
            outcome_training_result.total_loss
            if outcome_training_result is not None
            else None
        ),
        state_loss=(
            outcome_training_result.state_loss
            if outcome_training_result is not None
            else None
        ),
        event_loss=(
            outcome_training_result.event_loss
            if outcome_training_result is not None
            else None
        ),

        final_neural_state=(
            outcome_prediction.final_neural_state
        ),

        neural_state_mean=float(
            updated_outcome_neural_state.mean().item()
        ),
        neural_state_std=float(
            updated_outcome_neural_state.std(
                unbiased=False
            ).item()
        ),
        neural_state_min=float(
            updated_outcome_neural_state.min().item()
        ),
        neural_state_max=float(
            updated_outcome_neural_state.max().item()
        ),
    )

    goal_kind = (
        plan.kind.value
        if plan is not None
        else None
    )

    goal_target = (
        plan.target
        if plan is not None
        else None
    )

    termination_reason = info.get(
        "termination_reason"
    )

    metrics = StepMetrics(
        step=step,
        position=(agent.x, agent.y),

        energy=agent.energy,
        health=agent.health,
        curiosity=agent.curiosity,

        reward=reward,
        predicted_reward=(
            chosen_reward_prediction.predicted_reward
        ),
        loss=reward_loss,

        event=actual_event.name,
        visited_count=len(agent.visited),
        known_cell_count=len(agent.known_cells),

        choices_agree=choices_agree,

        grid_snapshot=info["grid"],

        rule_action=format_action(
            chosen_action
        ),
        action_reason=rule_choice.reason,

        network_action=format_action(
            network_choice.action
        ),

        termination_reason=termination_reason,
        goal_kind=goal_kind,
        goal_target=goal_target,

        outcome_model=outcome_metrics,
    )

    logger.debug(
        (
            "step=%d position=%s event=%s "
            "goal_kind=%s goal_target=%s "
            "rule_action=%s reason=%s "
            "network_action=%s reward=%.2f "
            "predicted_reward=%.2f "
            "reward_loss=%s "
            "visited=%d known=%d agree=%s "
            "termination_reason=%s"
        ),
        step,
        metrics.position,
        metrics.event,
        metrics.goal_kind,
        metrics.goal_target,
        metrics.rule_action,
        metrics.action_reason,
        metrics.network_action,
        metrics.reward,
        metrics.predicted_reward,
        metrics.loss,
        metrics.visited_count,
        metrics.known_cell_count,
        metrics.choices_agree,
        metrics.termination_reason,
    )

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
        outcome_metrics.predicted_energy_change,
        outcome_metrics.predicted_health_change,
        outcome_metrics.predicted_curiosity_change,

        outcome_metrics.actual_energy_change,
        outcome_metrics.actual_health_change,
        outcome_metrics.actual_curiosity_change,

        outcome_metrics.predicted_event,
        outcome_metrics.actual_event,
        outcome_metrics.event_correct,
        outcome_metrics.state_mae,
        outcome_metrics.reset_state_mae,
        outcome_metrics.persistent_better_than_reset,

        outcome_metrics.total_loss,
        outcome_metrics.state_loss,
        outcome_metrics.event_loss,
    )

    return (
        metrics,
        agreement_count,
        comparison_count,
        terminated,
        truncated,
        updated_outcome_neural_state.detach(),
    )