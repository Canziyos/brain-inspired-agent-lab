import random

import torch

from src.config import SimulationConfig
from src.core.world import DANGER, FOOD, MYSTERY, World
from src.learning.reward_network import ImmediateRewardNetwork


def create_world(
    config: SimulationConfig,
) -> World:
    world = World(
        width=config.world_width,
        height=config.world_height,
    )
    reserved = {(0, 0)}
    world_rng = random.Random(config.random_seed)

    world.place_random(
        FOOD,
        config.food_count,
        rng=world_rng,
        reserved=reserved,
    )

    world.place_random(
        DANGER,
        config.danger_count,
        rng=world_rng,
        reserved=reserved,
    )

    world.place_random(
        MYSTERY,
        config.mystery_count,
        rng=world_rng,
        reserved=reserved,
    )

    return world


def create_reward_network(
    config: SimulationConfig,
) -> tuple[
    ImmediateRewardNetwork,
    torch.optim.Optimizer,
]:
    reward_network = ImmediateRewardNetwork()

    optimizer = torch.optim.Adam(
        reward_network.parameters(),
        lr=config.learning_rate,
    )

    return reward_network, optimizer
