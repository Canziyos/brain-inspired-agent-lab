from collections.abc import Sequence

import torch

from src.learning.outcome.model import RateRecurrentOutcomeModel
from src.learning.samples import TransitionTrainingSample


def replay_neural_state(
    model: RateRecurrentOutcomeModel,
    samples: Sequence[TransitionTrainingSample],
    sequence_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    neural_state = model.initial_neural_state(
        batch_size=1,
        device=device,
        dtype=dtype,
    )

    recent_samples = samples[-sequence_length:]

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for sample in recent_samples:
            features = torch.tensor(
                sample.features,
                dtype=dtype,
                device=device,
            ).unsqueeze(0)

            _, _, neural_state = model.forward_with_state(
                features,
                neural_state,
            )

    if was_training:
        model.train()

    return neural_state.detach()