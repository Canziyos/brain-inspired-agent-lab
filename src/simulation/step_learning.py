import random
from dataclasses import dataclass

import torch

from src.core.dynamics_types import EventType
from src.learning.outcome import (
    OutcomeTrainingResult,
    RateRecurrentOutcomeModel,
    event_index,
    replay_neural_state,
    train_outcome_model,
)
from src.learning.reward_network import (
    ImmediateRewardNetwork,
    train_reward_network,
)
from src.learning.samples import (
    RewardTrainingSample,
    TransitionTrainingSample,
)
from src.learning.types import (
    FeatureVector,
    StateDelta,
)
from src.policies.neural_shadow import NetworkActionPrediction


@dataclass(frozen=True, slots=True)
class StepLearningResult:
    reward_loss: float | None
    outcome_training_result: OutcomeTrainingResult | None
    outcome_neural_state: torch.Tensor


def update_reward_learning(
    model: ImmediateRewardNetwork,
    optimizer: torch.optim.Optimizer,
    samples: list[RewardTrainingSample],
    prediction: NetworkActionPrediction,
    reward: float,
    batch_size: int,
    rng: random.Random,
) -> float | None:
    samples.append(
        RewardTrainingSample(
            features=prediction.features,
            reward=reward,
        )
    )

    return train_reward_network(
        model=model,
        optimizer=optimizer,
        reward_samples=samples,
        batch_size=batch_size,
        rng=rng,
    )


def update_outcome_learning(
    model: RateRecurrentOutcomeModel,
    optimizer: torch.optim.Optimizer,
    samples: list[TransitionTrainingSample],
    features: FeatureVector,
    state_delta: StateDelta,
    event: EventType,
    sequence_length: int,
    batch_size: int,
    rng: random.Random,
) -> OutcomeTrainingResult | None:
    samples.append(
        TransitionTrainingSample(
            features=features,
            state_delta=state_delta,
            event_index=event_index(event),
        )
    )

    return train_outcome_model(
        model=model,
        optimizer=optimizer,
        samples=samples,
        sequence_length=sequence_length,
        batch_size=batch_size,
        rng=rng,
    )


def refresh_outcome_neural_state(
    model: RateRecurrentOutcomeModel,
    samples: list[TransitionTrainingSample],
    predicted_neural_state: torch.Tensor,
    training_result: OutcomeTrainingResult | None,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if training_result is None:
        return predicted_neural_state

    return replay_neural_state(
        model=model,
        samples=samples,
        sequence_length=sequence_length,
        device=device,
        dtype=dtype,
    )


def update_step_learning(
    reward_network: ImmediateRewardNetwork,
    reward_optimizer: torch.optim.Optimizer,
    reward_samples: list[RewardTrainingSample],
    reward_prediction: NetworkActionPrediction,
    reward: float,
    reward_batch_size: int,
    reward_rng: random.Random,

    outcome_model: RateRecurrentOutcomeModel,
    outcome_optimizer: torch.optim.Optimizer,
    outcome_samples: list[TransitionTrainingSample],
    outcome_features: FeatureVector,
    outcome_state_delta: StateDelta,
    outcome_event: EventType,
    outcome_sequence_length: int,
    outcome_batch_size: int,
    outcome_rng: random.Random,

    predicted_outcome_neural_state: torch.Tensor,
    state_before_prediction: torch.Tensor,
) -> StepLearningResult:
    reward_loss = update_reward_learning(
        model=reward_network,
        optimizer=reward_optimizer,
        samples=reward_samples,
        prediction=reward_prediction,
        reward=reward,
        batch_size=reward_batch_size,
        rng=reward_rng,
    )

    outcome_training_result = update_outcome_learning(
        model=outcome_model,
        optimizer=outcome_optimizer,
        samples=outcome_samples,
        features=outcome_features,
        state_delta=outcome_state_delta,
        event=outcome_event,
        sequence_length=outcome_sequence_length,
        batch_size=outcome_batch_size,
        rng=outcome_rng,
    )

    outcome_neural_state = refresh_outcome_neural_state(
        model=outcome_model,
        samples=outcome_samples,
        predicted_neural_state=predicted_outcome_neural_state,
        training_result=outcome_training_result,
        sequence_length=outcome_sequence_length,
        device=state_before_prediction.device,
        dtype=state_before_prediction.dtype,
    )

    return StepLearningResult(
        reward_loss=reward_loss,
        outcome_training_result=outcome_training_result,
        outcome_neural_state=outcome_neural_state,
    )