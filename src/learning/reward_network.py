import random
from collections.abc import Sequence

import torch
from torch import nn

from src.learning.features import STATE_ACTION_FEATURE_COUNT
from src.learning.samples import RewardTrainingSample
from src.learning.types import FeatureVector


class ImmediateRewardNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_ACTION_FEATURE_COUNT, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def predict_reward(
    model: ImmediateRewardNetwork,
    features: FeatureVector,
) -> float:
    if len(features) != STATE_ACTION_FEATURE_COUNT:
        raise ValueError(
            "features must contain exactly "
            f"{STATE_ACTION_FEATURE_COUNT} values, got {len(features)}"
        )

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(feature_tensor)

    if was_training:
        model.train()

    return float(prediction.item())


def train_reward_network(
    model: ImmediateRewardNetwork,
    optimizer: torch.optim.Optimizer,
    reward_samples: Sequence[RewardTrainingSample],
    batch_size: int = 8,
    rng: random.Random | None = None,
) -> float | None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    if len(reward_samples) < batch_size:
        return None

    random_source = rng if rng is not None else random
    batch = random_source.sample(
        list(reward_samples),
        batch_size,
    )

    device = next(model.parameters()).device

    features = torch.tensor(
        [sample.features for sample in batch],
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.tensor(
        [sample.reward for sample in batch],
        dtype=torch.float32,
        device=device,
    )

    model.train()

    predictions = model(features)
    loss = nn.functional.mse_loss(predictions, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())