import torch

from src.configs.learning import LearningConfig
from src.configs.outcome import OutcomeConfig
from src.learning.outcome import RateRecurrentOutcomeModel
from src.learning.reward_network import ImmediateRewardNetwork


RewardModelBundle = tuple[
    ImmediateRewardNetwork,
    torch.optim.Optimizer,
]

OutcomeModelBundle = tuple[
    RateRecurrentOutcomeModel,
    torch.optim.Optimizer,
]


def create_reward_network(
    config: LearningConfig,
) -> RewardModelBundle:
    model = ImmediateRewardNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    return model, optimizer


def create_outcome_model(
    config: OutcomeConfig,
) -> OutcomeModelBundle:
    model = RateRecurrentOutcomeModel(
        neuron_count=config.neuron_count,
        neural_ticks=config.neural_ticks,
        leak=config.leak,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    return model, optimizer
