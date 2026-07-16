import logging
import random

import torch

from src.configs import SimulationConfig
from src.envs.grid_world_env import BabyViceGridEnv
from src.learning.outcome import RateRecurrentOutcomeModel
from src.learning.reward_network import ImmediateRewardNetwork
from src.learning.samples import (
    RewardTrainingSample,
    TransitionTrainingSample,
)
from src.memory.episode_trace import (
    EpisodicTrace,
    build_episode,
)
from src.memory.working_memory import WorkingMemory
from src.simulation.outcome_diagnostics import (
    build_outcome_model_metrics,
    predict_from_reset_state,
)
from src.simulation.step_decision import choose_step_decision
from src.simulation.step_execution import execute_step_action
from src.simulation.step_learning import update_step_learning
from src.simulation.step_logging import (
    log_outcome_metrics,
    log_step_metrics,
)
from src.simulation.step_metrics_builder import build_step_metrics
from src.telemetry.metrics import StepMetrics


logger = logging.getLogger(
    "brain_lab.simulation.step"
)


def run_simulation_step(
    step: int,
    config: SimulationConfig,
    env: BabyViceGridEnv,

    reward_network: ImmediateRewardNetwork,
    reward_optimizer: torch.optim.Optimizer,
    reward_samples: list[RewardTrainingSample],

    outcome_model: RateRecurrentOutcomeModel,
    outcome_optimizer: torch.optim.Optimizer,
    outcome_samples: list[TransitionTrainingSample],

    working_memory: WorkingMemory,
    episodic_trace: EpisodicTrace,

    policy_rng: random.Random,
    training_rng: random.Random,
    outcome_training_rng: random.Random,

    outcome_neural_state: torch.Tensor,
) -> tuple[StepMetrics, bool, bool, torch.Tensor]:
    if env.agent is None or env.world is None:
        raise RuntimeError(
            "Environment must be reset before simulation."
        )

    agent = env.agent
    world = env.world

    position_before = agent.position
    state_before = (
        float(agent.energy),
        float(agent.health),
        float(agent.curiosity),
    )

    decision = choose_step_decision(
        agent=agent,
        world=world,
        width=config.world.width,
        height=config.world.height,
        reward_network=reward_network,
        outcome_model=outcome_model,
        outcome_neural_state=outcome_neural_state,
        imagination_reward_weight=(
            config.imagination.reward_weight
        ),
        policy_rng=policy_rng,
        working_memory=working_memory,
    )

    reset_outcome_prediction = predict_from_reset_state(
        model=outcome_model,
        features=decision.chosen_imagination.features,
        reference_state=decision.state_before_prediction,
    )

    execution = execute_step_action(
        env=env,
        agent=agent,
        action=decision.rule_choice.action,
    )

    working_memory.record_step(
        agent=agent,
        action=decision.rule_choice.action,
        reward=execution.reward,
        event=execution.event,
    )
    working_memory.update_coverage(
        step=step,
        agent=agent,
        width=config.world.width,
        height=config.world.height,
    )

    episodic_trace.record(
        build_episode(
            step=step,
            position_before=position_before,
            state_before=state_before,
            plan=decision.plan,
            action=decision.rule_choice.action,
            reward=execution.reward,
            event=execution.event,
            position_after=agent.position,
            state_after=(
                float(agent.energy),
                float(agent.health),
                float(agent.curiosity),
            ),
            network_action=decision.network_choice.action,
            imagination_action=decision.imagined_choice.action,
            choices_agree=decision.choices_agree,
            imagination_agrees=decision.imagination_agrees,
        )
    )

    learning = update_step_learning(
        reward_network=reward_network,
        reward_optimizer=reward_optimizer,
        reward_samples=reward_samples,
        reward_prediction=(
            decision.chosen_reward_prediction
        ),
        reward=execution.reward,
        reward_batch_size=config.learning.batch_size,
        reward_rng=training_rng,

        outcome_model=outcome_model,
        outcome_optimizer=outcome_optimizer,
        outcome_samples=outcome_samples,
        outcome_features=(
            decision.chosen_imagination.features
        ),
        outcome_state_delta=execution.state_delta,
        outcome_event=execution.event,
        outcome_sequence_length=(
            config.outcome.sequence_length
        ),
        outcome_batch_size=(
            config.outcome.sequence_batch_size
        ),
        outcome_rng=outcome_training_rng,

        predicted_outcome_neural_state=(
            decision.chosen_imagination.next_neural_state
        ),
        state_before_prediction=(
            decision.state_before_prediction
        ),
    )

    outcome_metrics = build_outcome_model_metrics(
        prediction=decision.chosen_imagination.prediction,
        reset_prediction=reset_outcome_prediction,
        actual_event=execution.event,
        actual_state_delta=execution.state_delta,
        training_result=learning.outcome_training_result,
        neural_state=learning.outcome_neural_state,
    )

    metrics = build_step_metrics(
        step=step,
        agent=agent,
        decision=decision,
        execution=execution,
        learning=learning,
        outcome_metrics=outcome_metrics,
        memory=working_memory.snapshot(),
    )

    log_step_metrics(
        logger=logger,
        metrics=metrics,
    )

    log_outcome_metrics(
        logger=logger,
        metrics=outcome_metrics,
    )

    return (
        metrics,
        execution.terminated,
        execution.truncated,
        learning.outcome_neural_state.detach(),
    )
