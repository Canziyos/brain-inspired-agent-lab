from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.memory.episode_trace import Episode
from src.planning.goal_planner import GoalPlan


Position = tuple[int, int]

STATE_ENERGY_SCALE = 50.0
STATE_HEALTH_SCALE = 100.0
STATE_CURIOSITY_SCALE = 50.0
SIMILARITY_PRIOR_WEIGHT = 3.0
CONFIDENCE_MATCH_SCALE = 8.0


@dataclass(frozen=True, slots=True)
class EpisodeQuery:
    position: Position
    energy: float
    health: float
    curiosity: float
    goal_kind: str | None
    goal_id: str | None
    candidate_actions: frozenset[Action]


@dataclass(frozen=True, slots=True)
class ActionEpisodeStats:
    action: Action
    match_count: int
    total_weight: float
    expected_reward: float
    best_event: EventType | None
    risk_hit_danger: float


@dataclass(frozen=True, slots=True)
class EpisodicActionAdvice:
    action: Action | None
    expected_reward: float
    confidence: float
    match_count: int
    best_event: EventType | None
    risk_hit_danger: float
    rationale: str

    @property
    def has_advice(self) -> bool:
        return self.action is not None


NO_EPISODIC_ADVICE = EpisodicActionAdvice(
    action=None,
    expected_reward=0.0,
    confidence=0.0,
    match_count=0,
    best_event=None,
    risk_hit_danger=0.0,
    rationale="no similar episodes yet",
)


def advise_from_episodes(
    agent: Agent,
    plan: GoalPlan | None,
    evaluations: Sequence[ActionEvaluation],
    episodes: Sequence[Episode],
) -> EpisodicActionAdvice:
    query = build_episode_query(
        agent=agent,
        plan=plan,
        evaluations=evaluations,
    )

    if not query.candidate_actions or not episodes:
        return NO_EPISODIC_ADVICE

    action_stats = action_episode_stats(
        query=query,
        episodes=episodes,
    )

    if not action_stats:
        return NO_EPISODIC_ADVICE

    ranked_stats = sorted(
        action_stats,
        key=lambda stats: (
            stats.expected_reward,
            stats.total_weight,
            stats.match_count,
            stats.action.name,
        ),
        reverse=True,
    )

    best = ranked_stats[0]
    second_expected_reward = (
        ranked_stats[1].expected_reward
        if len(ranked_stats) > 1
        else None
    )

    return EpisodicActionAdvice(
        action=best.action,
        expected_reward=best.expected_reward,
        confidence=advice_confidence(
            best=best,
            second_expected_reward=second_expected_reward,
        ),
        match_count=best.match_count,
        best_event=best.best_event,
        risk_hit_danger=best.risk_hit_danger,
        rationale=advice_rationale(best),
    )


def build_episode_query(
    agent: Agent,
    plan: GoalPlan | None,
    evaluations: Sequence[ActionEvaluation],
) -> EpisodeQuery:
    return EpisodeQuery(
        position=agent.position,
        energy=float(agent.energy),
        health=float(agent.health),
        curiosity=float(agent.curiosity),
        goal_kind=(
            plan.kind.value
            if plan is not None
            else None
        ),
        goal_id=(
            plan.goal_id
            if plan is not None
            else None
        ),
        candidate_actions=frozenset(
            evaluation.action
            for evaluation in evaluations
        ),
    )


def action_episode_stats(
    query: EpisodeQuery,
    episodes: Sequence[Episode],
) -> tuple[ActionEpisodeStats, ...]:
    weighted_rewards: dict[Action, float] = defaultdict(float)
    total_weights: dict[Action, float] = defaultdict(float)
    counts: dict[Action, int] = defaultdict(int)
    event_weights: dict[Action, dict[EventType, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    danger_weights: dict[Action, float] = defaultdict(float)

    for episode in episodes:
        if episode.action not in query.candidate_actions:
            continue

        weight = episode_similarity_weight(
            query=query,
            episode=episode,
        )

        if weight <= 0.0:
            continue

        weighted_rewards[episode.action] += episode.reward * weight
        total_weights[episode.action] += weight
        counts[episode.action] += 1
        event_weights[episode.action][episode.event] += weight

        if episode.event is EventType.HIT_DANGER:
            danger_weights[episode.action] += weight

    stats: list[ActionEpisodeStats] = []

    for action, total_weight in total_weights.items():
        if total_weight <= 0.0:
            continue

        best_event = weighted_mode(event_weights[action])

        stats.append(
            ActionEpisodeStats(
                action=action,
                match_count=counts[action],
                total_weight=total_weight,
                expected_reward=(
                    weighted_rewards[action] / total_weight
                ),
                best_event=best_event,
                risk_hit_danger=(
                    danger_weights[action] / total_weight
                ),
            )
        )

    return tuple(stats)


def episode_similarity_weight(
    query: EpisodeQuery,
    episode: Episode,
) -> float:
    goal_weight = goal_similarity_weight(
        query_goal_kind=query.goal_kind,
        query_goal_id=query.goal_id,
        episode_goal_kind=episode.goal_kind,
        episode_goal_id=episode.goal_id,
    )

    if goal_weight <= 0.0:
        return 0.0

    distance_penalty = chebyshev_distance(
        query.position,
        episode.position_before,
    )
    state_penalty = state_distance(
        query=(query.energy, query.health, query.curiosity),
        episode=episode.state_before,
    )

    return goal_weight / (
        1.0
        + distance_penalty
        + state_penalty
    )


def goal_similarity_weight(
    query_goal_kind: str | None,
    query_goal_id: str | None,
    episode_goal_kind: str | None,
    episode_goal_id: str | None,
) -> float:
    if (
        query_goal_id is not None
        and episode_goal_id is not None
        and query_goal_id == episode_goal_id
    ):
        return 1.0

    if (
        query_goal_kind is not None
        and episode_goal_kind is not None
        and query_goal_kind == episode_goal_kind
    ):
        return 0.6

    if query_goal_kind is None or episode_goal_kind is None:
        return 0.3

    return 0.15


def state_distance(
    query: tuple[float, float, float],
    episode: tuple[float, float, float],
) -> float:
    query_energy, query_health, query_curiosity = query
    episode_energy, episode_health, episode_curiosity = episode

    return (
        abs(query_energy - episode_energy) / STATE_ENERGY_SCALE
        + abs(query_health - episode_health) / STATE_HEALTH_SCALE
        + abs(query_curiosity - episode_curiosity) / STATE_CURIOSITY_SCALE
    )


def chebyshev_distance(
    first: Position,
    second: Position,
) -> int:
    return max(
        abs(first[0] - second[0]),
        abs(first[1] - second[1]),
    )


def weighted_mode(
    weights: dict[EventType, float],
) -> EventType | None:
    if not weights:
        return None

    return max(
        weights,
        key=lambda event: (weights[event], event.value),
    )


def advice_confidence(
    best: ActionEpisodeStats,
    second_expected_reward: float | None,
) -> float:
    weight_confidence = best.total_weight / (
        best.total_weight + SIMILARITY_PRIOR_WEIGHT
    )
    match_confidence = min(
        1.0,
        best.match_count / CONFIDENCE_MATCH_SCALE,
    )

    if second_expected_reward is None:
        separation_confidence = 0.5
    else:
        margin = best.expected_reward - second_expected_reward
        scale = (
            abs(best.expected_reward)
            + abs(second_expected_reward)
            + 1.0
        )
        separation_confidence = max(
            0.0,
            min(1.0, margin / scale),
        )

    return max(
        0.0,
        min(
            1.0,
            0.5 * weight_confidence
            + 0.3 * match_confidence
            + 0.2 * separation_confidence,
        ),
    )


def advice_rationale(stats: ActionEpisodeStats) -> str:
    event_text = (
        stats.best_event.value
        if stats.best_event is not None
        else "unknown"
    )

    return (
        f"episodic advisor: {stats.match_count} similar "
        f"episodes, expected_reward={stats.expected_reward:.3f}, "
        f"common_event={event_text}, "
        f"danger_risk={stats.risk_hit_danger:.3f}"
    )
