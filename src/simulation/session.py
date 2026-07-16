import random
from dataclasses import dataclass

import torch

from src.configs import SimulationConfig
from src.envs.grid_world_env import BabyViceGridEnv
from src.learning.model_factory import (
    create_outcome_model,
    create_reward_network,
)
from src.learning.outcome import RateRecurrentOutcomeModel
from src.learning.reward_network import ImmediateRewardNetwork
from src.learning.samples import (
    RewardTrainingSample,
    TransitionTrainingSample,
)
from src.memory.episode_store import LoadedEpisodeStore, load_episode_store
from src.memory.episode_trace import Episode, EpisodicTrace
from src.memory.working_memory import WorkingMemory


@dataclass(slots=True)
class SimulationSession:
    env: BabyViceGridEnv

    reward_network: ImmediateRewardNetwork
    reward_optimizer: torch.optim.Optimizer
    reward_samples: list[RewardTrainingSample]

    outcome_model: RateRecurrentOutcomeModel
    outcome_optimizer: torch.optim.Optimizer
    outcome_samples: list[TransitionTrainingSample]
    outcome_neural_state: torch.Tensor

    working_memory: WorkingMemory
    episodic_trace: EpisodicTrace
    prior_episodes: tuple[Episode, ...]
    loaded_episode_store: LoadedEpisodeStore

    policy_rng: random.Random
    reward_training_rng: random.Random
    outcome_training_rng: random.Random



def create_simulation_session(
    config: SimulationConfig,
) -> SimulationSession:
    policy_rng = random.Random(
        config.runtime.random_seed + 1
    )

    reward_training_rng = random.Random(
        config.runtime.random_seed + 2
    )

    outcome_training_rng = random.Random(
        config.runtime.random_seed + 3
    )

    loaded_episode_store = load_episode_store(config)

    env = BabyViceGridEnv(config=config)
    env.reset(seed=config.runtime.random_seed)

    torch.manual_seed(config.runtime.torch_seed)

    reward_network, reward_optimizer = create_reward_network(
        config.learning
    )

    torch.manual_seed(config.runtime.torch_seed + 1)

    outcome_model, outcome_optimizer = create_outcome_model(
        config.outcome
    )

    outcome_neural_state = outcome_model.initial_neural_state(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    return SimulationSession(
        env=env,
        reward_network=reward_network,
        reward_optimizer=reward_optimizer,
        reward_samples=[],
        outcome_model=outcome_model,
        outcome_optimizer=outcome_optimizer,
        outcome_samples=[],
        outcome_neural_state=outcome_neural_state,
        working_memory=WorkingMemory(),
        episodic_trace=EpisodicTrace(),
        prior_episodes=loaded_episode_store.episodes,
        loaded_episode_store=loaded_episode_store,
        policy_rng=policy_rng,
        reward_training_rng=reward_training_rng,
        outcome_training_rng=outcome_training_rng,
    )
