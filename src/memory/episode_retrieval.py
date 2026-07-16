from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
MAX_SIMILAR_EPISODES_PER_ACTION = 8
MIN_EPISODE_SIMILARITY_WEIGHT = 0.08
MIN_RELATIVE_SIMILARITY_WEIGHT = 0.35

MIN_USABLE_MATCH_COUNT = 3
MIN_USABLE_CONFIDENCE = 0.35
MIN_USABLE_RELIABILITY = 0.35
MIN_USABLE_TOTAL_WEIGHT = 0.50
MAX_USABLE_DANGER_RISK = 0.75

REWARD_SHRINKAGE_PRIOR = 2.0
RARE_EVENT_MIN_COUNT = 3
RARE_EVENT_DAMPING = 0.45
RARE_REWARD_EVENTS = frozenset(
    {
        EventType.ATE_FOOD,
        EventType.DISCOVERED_MYSTERY,
    }
)


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
class WeightedEpisodeMatch:
    episode: Episode
    weight: float


@dataclass(frozen=True, slots=True)
class ActionEpisodeStats:
    action: Action
    match_count: int
    total_weight: float
    raw_expected_reward: float
    expected_reward: float
    best_event: EventType | None
    best_event_count: int
    risk_hit_danger: float
    rare_event_dampened: bool


@dataclass(frozen=True, slots=True)
class EpisodicActionAdvice:
    action: Action | None
    raw_expected_reward: float
    expected_reward: float
    confidence: float
    reliability: float
    match_count: int
    best_event: EventType | None
    risk_hit_danger: float
    is_usable: bool
    reliability_reason: str
    rationale: str

    @property
    def has_advice(self) -> bool:
        return self.action is not None


NO_EPISODIC_ADVICE = EpisodicActionAdvice(
    action=None,
    raw_expected_reward=0.0,
    expected_reward=0.0,
    confidence=0.0,
    reliability=0.0,
    match_count=0,
    best_event=None,
    risk_hit_danger=0.0,
    is_usable=False,
    reliability_reason="no similar episodes yet",
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
    confidence = advice_confidence(
        best=best,
        second_expected_reward=second_expected_reward,
    )
    reliability = advice_reliability(
        best=best,
        confidence=confidence,
    )
    reliability_reason = advice_reliability_reason(
        best=best,
        confidence=confidence,
        reliability=reliability,
    )
    is_usable = advice_is_usable(
        best=best,
        confidence=confidence,
        reliability=reliability,
    )

    return EpisodicActionAdvice(
        action=best.action,
        raw_expected_reward=best.raw_expected_reward,
        expected_reward=best.expected_reward,
        confidence=confidence,
        reliability=reliability,
        match_count=best.match_count,
        best_event=best.best_event,
        risk_hit_danger=best.risk_hit_danger,
        is_usable=is_usable,
        reliability_reason=reliability_reason,
        rationale=advice_rationale(
            stats=best,
            confidence=confidence,
            reliability=reliability,
            is_usable=is_usable,
            reliability_reason=reliability_reason,
        ),
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
    matches_by_action = collect_top_similar_matches(
        query=query,
        episodes=episodes,
    )

    weighted_rewards: dict[Action, float] = defaultdict(float)
    total_weights: dict[Action, float] = defaultdict(float)
    counts: dict[Action, int] = defaultdict(int)
    event_weights: dict[Action, dict[EventType, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    event_counts: dict[Action, dict[EventType, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    danger_weights: dict[Action, float] = defaultdict(float)

    for action, matches in matches_by_action.items():
        for match in matches:
            episode = match.episode
            weight = match.weight

            weighted_rewards[action] += episode.reward * weight
            total_weights[action] += weight
            counts[action] += 1
            event_weights[action][episode.event] += weight
            event_counts[action][episode.event] += 1

            if episode.event is EventType.HIT_DANGER:
                danger_weights[action] += weight

    stats: list[ActionEpisodeStats] = []

    for action, total_weight in total_weights.items():
        if total_weight <= 0.0:
            continue

        best_event = weighted_mode(event_weights[action])
        best_event_count = (
            event_counts[action][best_event]
            if best_event is not None
            else 0
        )
        raw_expected_reward = weighted_rewards[action] / total_weight
        expected_reward = calibrated_expected_reward(
            raw_expected_reward=raw_expected_reward,
            total_weight=total_weight,
            best_event=best_event,
            best_event_count=best_event_count,
        )
        rare_event_dampened = rare_event_was_dampened(
            raw_expected_reward=raw_expected_reward,
            best_event=best_event,
            best_event_count=best_event_count,
        )

        stats.append(
            ActionEpisodeStats(
                action=action,
                match_count=counts[action],
                total_weight=total_weight,
                raw_expected_reward=raw_expected_reward,
                expected_reward=expected_reward,
                best_event=best_event,
                best_event_count=best_event_count,
                risk_hit_danger=(
                    danger_weights[action] / total_weight
                ),
                rare_event_dampened=rare_event_dampened,
            )
        )

    return tuple(stats)


def collect_top_similar_matches(
    query: EpisodeQuery,
    episodes: Sequence[Episode],
) -> dict[Action, tuple[WeightedEpisodeMatch, ...]]:
    matches_by_action: dict[
        Action,
        list[WeightedEpisodeMatch],
    ] = defaultdict(list)

    for episode in episodes:
        if episode.action not in query.candidate_actions:
            continue

        weight = episode_similarity_weight(
            query=query,
            episode=episode,
        )

        if weight <= 0.0:
            continue

        matches_by_action[episode.action].append(
            WeightedEpisodeMatch(
                episode=episode,
                weight=weight,
            )
        )

    selected_by_action: dict[Action, tuple[WeightedEpisodeMatch, ...]] = {}

    for action, matches in matches_by_action.items():
        ranked_matches = sorted(
            matches,
            key=lambda match: (
                match.weight,
                match.episode.step,
            ),
            reverse=True,
        )
        if not ranked_matches:
            continue

        strongest_weight = ranked_matches[0].weight
        similarity_floor = max(
            MIN_EPISODE_SIMILARITY_WEIGHT,
            strongest_weight * MIN_RELATIVE_SIMILARITY_WEIGHT,
        )
        selected = tuple(
            match
            for match in ranked_matches
            if match.weight >= similarity_floor
        )[:MAX_SIMILAR_EPISODES_PER_ACTION]

        if selected:
            selected_by_action[action] = selected

    return selected_by_action


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


def calibrated_expected_reward(
    raw_expected_reward: float,
    total_weight: float,
    best_event: EventType | None,
    best_event_count: int,
) -> float:
    evidence_weight = total_weight / (
        total_weight + REWARD_SHRINKAGE_PRIOR
    )
    calibrated = raw_expected_reward * evidence_weight

    if rare_event_was_dampened(
        raw_expected_reward=raw_expected_reward,
        best_event=best_event,
        best_event_count=best_event_count,
    ):
        calibrated *= RARE_EVENT_DAMPING

    return calibrated


def rare_event_was_dampened(
    raw_expected_reward: float,
    best_event: EventType | None,
    best_event_count: int,
) -> bool:
    return (
        raw_expected_reward > 0.0
        and best_event in RARE_REWARD_EVENTS
        and best_event_count < RARE_EVENT_MIN_COUNT
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

    return clamp01(
        0.5 * weight_confidence
        + 0.3 * match_confidence
        + 0.2 * separation_confidence
    )


def advice_reliability(
    best: ActionEpisodeStats,
    confidence: float,
) -> float:
    match_signal = min(
        1.0,
        best.match_count / MIN_USABLE_MATCH_COUNT,
    )
    weight_signal = best.total_weight / (
        best.total_weight + SIMILARITY_PRIOR_WEIGHT
    )
    safety_signal = 1.0 - min(1.0, best.risk_hit_danger)
    rare_event_penalty = 0.15 if best.rare_event_dampened else 0.0

    return clamp01(
        0.40 * confidence
        + 0.30 * match_signal
        + 0.20 * weight_signal
        + 0.10 * safety_signal
        - rare_event_penalty
    )


def advice_is_usable(
    best: ActionEpisodeStats,
    confidence: float,
    reliability: float,
) -> bool:
    return (
        best.match_count >= MIN_USABLE_MATCH_COUNT
        and best.total_weight >= MIN_USABLE_TOTAL_WEIGHT
        and confidence >= MIN_USABLE_CONFIDENCE
        and reliability >= MIN_USABLE_RELIABILITY
        and best.risk_hit_danger <= MAX_USABLE_DANGER_RISK
    )


def advice_reliability_reason(
    best: ActionEpisodeStats,
    confidence: float,
    reliability: float,
) -> str:
    reasons: list[str] = []

    if best.match_count < MIN_USABLE_MATCH_COUNT:
        reasons.append("low_match_count")

    if best.total_weight < MIN_USABLE_TOTAL_WEIGHT:
        reasons.append("low_similarity_weight")

    if confidence < MIN_USABLE_CONFIDENCE:
        reasons.append("low_confidence")

    if reliability < MIN_USABLE_RELIABILITY:
        reasons.append("low_reliability")

    if best.risk_hit_danger > MAX_USABLE_DANGER_RISK:
        reasons.append("high_danger_risk")

    if best.rare_event_dampened:
        reasons.append("rare_reward_event_dampened")

    if not reasons:
        return "usable"

    return ";".join(reasons)


def advice_rationale(
    stats: ActionEpisodeStats,
    confidence: float,
    reliability: float,
    is_usable: bool,
    reliability_reason: str,
) -> str:
    event_text = (
        stats.best_event.value
        if stats.best_event is not None
        else "unknown"
    )
    status = "usable" if is_usable else "weak"

    return (
        f"episodic advisor ({status}): {stats.match_count} similar "
        f"episodes used, raw_expected_reward={stats.raw_expected_reward:.3f}, "
        f"calibrated_expected_reward={stats.expected_reward:.3f}, "
        f"confidence={confidence:.3f}, reliability={reliability:.3f}, "
        f"common_event={event_text}, "
        f"danger_risk={stats.risk_hit_danger:.3f}, "
        f"max_matches_per_action={MAX_SIMILAR_EPISODES_PER_ACTION}, "
        f"min_similarity_weight={MIN_EPISODE_SIMILARITY_WEIGHT:.3f}, "
        f"min_relative_similarity={MIN_RELATIVE_SIMILARITY_WEIGHT:.3f}, "
        f"reason={reliability_reason}"
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))