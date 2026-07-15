import random
from collections.abc import Sequence

import torch
from torch import nn

from src.learning.outcome.events import EVENT_TYPES
from src.learning.outcome.model import RateRecurrentOutcomeModel
from src.learning.outcome.types import OutcomeTrainingResult
from src.learning.samples import TransitionTrainingSample


def train_outcome_model(
    model: RateRecurrentOutcomeModel,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[TransitionTrainingSample],
    sequence_length: int = 8,
    batch_size: int = 4,
    rng: random.Random | None = None,
) -> OutcomeTrainingResult | None:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if len(samples) < sequence_length:
        return None

    windows = _sample_windows(
        samples=samples,
        sequence_length=sequence_length,
        batch_size=batch_size,
        rng=rng,
    )

    device = next(model.parameters()).device

    features = torch.tensor(
        [
            [sample.features for sample in window]
            for window in windows
        ],
        dtype=torch.float32,
        device=device,
    )

    state_deltas = torch.tensor(
        [
            [sample.state_delta for sample in window]
            for window in windows
        ],
        dtype=torch.float32,
        device=device,
    )

    event_indices = torch.tensor(
        [
            [sample.event_index for sample in window]
            for window in windows
        ],
        dtype=torch.long,
        device=device,
    )

    neural_state = model.initial_neural_state(
        batch_size=batch_size,
        device=device,
        dtype=features.dtype,
    )

    predicted_deltas: list[torch.Tensor] = []
    predicted_event_logits: list[torch.Tensor] = []

    model.train()

    for time_index in range(sequence_length):
        step_deltas, step_event_logits, neural_state = (
            model.forward_with_state(
                features[:, time_index, :],
                neural_state,
            )
        )

        predicted_deltas.append(step_deltas)
        predicted_event_logits.append(step_event_logits)

    predicted_delta_tensor = torch.stack(
        predicted_deltas,
        dim=1,
    )

    predicted_event_tensor = torch.stack(
        predicted_event_logits,
        dim=1,
    )

    state_loss = nn.functional.mse_loss(
        predicted_delta_tensor,
        state_deltas,
    )

    event_loss = nn.functional.cross_entropy(
        predicted_event_tensor.reshape(-1, len(EVENT_TYPES)),
        event_indices.reshape(-1),
    )

    total_loss = state_loss + event_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return OutcomeTrainingResult(
        total_loss=float(total_loss.item()),
        state_loss=float(state_loss.item()),
        event_loss=float(event_loss.item()),
    )


def _sample_windows(
    samples: Sequence[TransitionTrainingSample],
    sequence_length: int,
    batch_size: int,
    rng: random.Random | None,
) -> list[Sequence[TransitionTrainingSample]]:
    random_source = rng if rng is not None else random
    max_start = len(samples) - sequence_length

    starts = [
        random_source.randint(0, max_start)
        for _ in range(batch_size)
    ]

    return [
        samples[start:start + sequence_length]
        for start in starts
    ]