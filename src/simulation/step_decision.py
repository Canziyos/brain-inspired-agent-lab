import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.core.actions import ActionEvaluation
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import World
from src.learning.outcome import RateRecurrentOutcomeModel
from src.learning.reward_network import ImmediateRewardNetwork
from src.memory.episode_retrieval import (
    EpisodicActionAdvice,
    advise_from_episodes,
)
from src.memory.episode_trace import Episode
from src.memory.working_memory import WorkingMemory
from src.planning.goal_planner import (
    GoalPlan,
    select_goal_plan,
)
from src.planning.imagination import (
    ImaginedAction,
    choose_imagined_action,
    find_imagined_action,
    imagine_actions,
)
from src.policies.neural_shadow import (
    NetworkActionPrediction,
    choose_network_action,
    find_action_prediction,
    predict_network_actions,
)
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)


@dataclass(frozen=True, slots=True)
class StepDecision:
    observations: tuple[Observation, ...]
    plan: GoalPlan | None

    evaluations: tuple[ActionEvaluation, ...]

    rule_choice: ActionEvaluation

    network_choice: NetworkActionPrediction
    chosen_reward_prediction: NetworkActionPrediction
    choices_agree: bool

    imagined_choice: ImaginedAction
    chosen_imagination: ImaginedAction
    imagination_agrees: bool

    episodic_advice: EpisodicActionAdvice
    prior_episodic_advice: EpisodicActionAdvice
    prior_episode_count: int
    same_run_episode_count: int

    state_before_prediction: torch.Tensor



def choose_step_decision(
    agent: Agent,
    world: World,
    width: int,
    height: int,
    reward_network: ImmediateRewardNetwork,
    outcome_model: RateRecurrentOutcomeModel,
    outcome_neural_state: torch.Tensor,
    imagination_reward_weight: float,
    policy_rng: random.Random,
    working_memory: WorkingMemory,
    prior_episodic_episodes: Sequence[Episode],
    same_run_episodic_episodes: Sequence[Episode],
) -> StepDecision:
    observations = tuple(
        agent.sense(world)
    )

    agent.observe(observations)

    plan = select_goal_plan(
        agent=agent,
        width=width,
        height=height,
        preference=working_memory.goal_preference(),
    )

    working_memory.remember_selected_goal(plan)

    evaluations = tuple(
        evaluate_actions(
            agent=agent,
            observations=observations,
            plan=plan,
        )
    )

    combined_episodes = tuple(prior_episodic_episodes) + tuple(
        same_run_episodic_episodes
    )

    episodic_advice = advise_from_episodes(
        agent=agent,
        plan=plan,
        evaluations=evaluations,
        episodes=combined_episodes,
    )

    prior_episodic_advice = advise_from_episodes(
        agent=agent,
        plan=plan,
        evaluations=evaluations,
        episodes=prior_episodic_episodes,
    )

    rule_choice = choose_action(
        evaluations=evaluations,
        rng=policy_rng,
    )

    state_before_prediction = outcome_neural_state.detach()

    imagined_actions = imagine_actions(
        agent=agent,
        evaluations=evaluations,
        model=outcome_model,
        neural_state=state_before_prediction,
        reward_weight=imagination_reward_weight,
    )

    imagined_choice = choose_imagined_action(
        imagined_actions
    )

    chosen_imagination = find_imagined_action(
        chosen_action=rule_choice.action,
        imagined_actions=imagined_actions,
    )

    reward_predictions = predict_network_actions(
        agent=agent,
        evaluations=evaluations,
        model=reward_network,
    )

    network_choice = choose_network_action(
        reward_predictions
    )

    chosen_reward_prediction = find_action_prediction(
        chosen_action=rule_choice.action,
        predictions=reward_predictions,
    )

    return StepDecision(
        observations=observations,
        plan=plan,
        evaluations=evaluations,
        rule_choice=rule_choice,
        network_choice=network_choice,
        chosen_reward_prediction=chosen_reward_prediction,
        choices_agree=(
            rule_choice.action is network_choice.action
        ),
        imagined_choice=imagined_choice,
        chosen_imagination=chosen_imagination,
        imagination_agrees=(
            rule_choice.action is imagined_choice.action
        ),
        episodic_advice=episodic_advice,
        prior_episodic_advice=prior_episodic_advice,
        prior_episode_count=len(prior_episodic_episodes),
        same_run_episode_count=len(same_run_episodic_episodes),
        state_before_prediction=state_before_prediction,
    )
