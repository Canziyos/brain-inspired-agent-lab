import random

import torch
from torch import nn

from src.learning.samples import RewardSample

from src.learning.features import (
    STATE_ACTION_FEATURE_COUNT,
)


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
    features: tuple[float, ...],
) -> float:
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(feature_tensor)

    return prediction.item()


def train_reward_network(
    model: ImmediateRewardNetwork,
    optimizer: torch.optim.Optimizer,
    reward_samples: list[RewardSample],
    batch_size: int = 8,
    rng: random.Random | None = None,
) -> float | None:
    if len(reward_samples) < batch_size:
        return None

    random_source = rng if rng is not None else random
    batch = random_source.sample(reward_samples, batch_size)

    features = torch.tensor(
        [reward_sample.features for reward_sample in batch],
        dtype=torch.float32,
    )

    rewards = torch.tensor(
        [reward_sample.reward for reward_sample in batch],
        dtype=torch.float32,
    )

    model.train()

    predictions = model(features)
    loss = nn.functional.mse_loss(predictions, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
