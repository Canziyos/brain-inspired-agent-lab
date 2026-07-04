import logging
import random

import torch

from src.config import SimulationConfig
from src.core.actions import Action
from src.envs.grid_world_env import (
    BabyViceGridEnv,
    action_to_discrete,
)
from src.learning.reward_network import (
    ImmediateRewardNetwork,
    train_reward_network,
)
from src.learning.samples import RewardSample
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
from src.simulation.metrics import StepMetrics


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
    optimizer: torch.optim.Optimizer,
    reward_samples: list[RewardSample],
    policy_rng: random.Random,
    training_rng: random.Random,
    agreement_count: int,
    comparison_count: int,
) -> tuple[StepMetrics, int, int, bool, bool]:
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

    predictions = predict_actions(
        agent=agent,
        evaluations=evaluations,
        model=reward_network,
    )

    network_choice = choose_network_action(
        predictions
    )

    chosen_prediction = find_action_prediction(
        chosen_action=rule_choice.action,
        predictions=predictions,
    )

    choices_agree = actions_match(
        rule_choice.action,
        network_choice.action,
    )

    comparison_count += 1

    if choices_agree:
        agreement_count += 1

    discrete_action = action_to_discrete(
        position=(agent.x, agent.y),
        action=rule_choice.action,
    )

    (
        _observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(discrete_action)

    reward_samples.append(
        RewardSample(
            features=chosen_prediction.features,
            reward=reward,
        )
    )

    loss = train_reward_network(
        model=reward_network,
        optimizer=optimizer,
        reward_samples=reward_samples,
        batch_size=config.batch_size,
        rng=training_rng,
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
            chosen_prediction.predicted_reward
        ),
        loss=loss,

        event=info["event"],
        visited_count=len(agent.visited),
        known_cell_count=len(agent.known_cells),

        choices_agree=choices_agree,

        grid_snapshot=info["grid"],

        rule_action=format_action(
            rule_choice.action
        ),
        action_reason=rule_choice.reason,

        network_action=format_action(
            network_choice.action
        ),

        termination_reason=termination_reason,
        goal_kind=goal_kind,
        goal_target=goal_target,
    )

    logger.debug(
        (
            "step=%d position=%s event=%s "
            "goal_kind=%s goal_target=%s "
            "rule_action=%s reason=%s "
            "network_action=%s reward=%.2f "
            "predicted_reward=%.2f loss=%s "
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

    return (
        metrics,
        agreement_count,
        comparison_count,
        terminated,
        truncated,
    )