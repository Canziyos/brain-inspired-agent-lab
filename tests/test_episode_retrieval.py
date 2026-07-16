from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.memory.episode_retrieval import (
    MAX_SIMILAR_EPISODES_PER_ACTION,
    MIN_EPISODE_SIMILARITY_WEIGHT,
    MIN_RELATIVE_SIMILARITY_WEIGHT,
    NO_EPISODIC_ADVICE,
    action_episode_stats,
    advise_from_episodes,
    build_episode_query,
)
from src.memory.episode_trace import Episode
from src.planning.goal_planner import GoalKind, GoalPlan


def episode(
    action: Action,
    reward: float,
    event: EventType = EventType.VISITED_EMPTY,
    position_before: tuple[int, int] = (1, 1),
    energy: float = 50.0,
    curiosity: float = 20.0,
    goal_id: str | None = "frontier:1:1",
    step: int = 0,
) -> Episode:
    return Episode(
        step=step,
        position_before=position_before,
        state_before=(energy, 100.0, curiosity),
        goal_kind="frontier",
        goal_target=(2, 2),
        goal_id=goal_id,
        action=action,
        reward=reward,
        event=event,
        position_after=(2, 2),
        state_after=(energy - 2.0, 100.0, curiosity),
        network_action=Action.REST,
        imagination_action=action,
        choices_agree=False,
        imagination_agrees=True,
    )


def plan() -> GoalPlan:
    return GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(2, 2),
        path=((1, 1), (2, 2)),
        goal_id="frontier:1:1",
    )


def evaluations() -> tuple[ActionEvaluation, ...]:
    return (
        ActionEvaluation(Action.MOVE_EAST, 10.0),
        ActionEvaluation(Action.MOVE_NORTH, 10.0),
    )


def agent() -> Agent:
    return Agent(x=1, y=1, energy=50.0, curiosity=20.0)


def test_no_episodes_produces_no_advice() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=evaluations(),
        episodes=(),
    )

    assert advice == NO_EPISODIC_ADVICE
    assert not advice.has_advice
    assert not advice.is_usable


def test_episodic_advisor_prefers_action_with_better_memory() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=evaluations(),
        episodes=(
            episode(Action.MOVE_EAST, reward=-2.0),
            episode(Action.MOVE_EAST, reward=-1.0),
            episode(Action.MOVE_NORTH, reward=3.0),
            episode(Action.MOVE_NORTH, reward=2.0),
        ),
    )

    assert advice.action is Action.MOVE_NORTH
    assert advice.raw_expected_reward > 0.0
    assert advice.expected_reward > 0.0
    assert advice.expected_reward < advice.raw_expected_reward
    assert advice.match_count == 2
    assert advice.confidence > 0.0
    assert advice.has_advice
    assert not advice.is_usable
    assert "low_match_count" in advice.reliability_reason


def test_episodic_advisor_marks_enough_evidence_as_usable() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=evaluations(),
        episodes=(
            episode(Action.MOVE_EAST, reward=-2.0),
            episode(Action.MOVE_EAST, reward=-1.0),
            episode(Action.MOVE_NORTH, reward=3.0),
            episode(Action.MOVE_NORTH, reward=2.0),
            episode(Action.MOVE_NORTH, reward=2.5),
            episode(Action.MOVE_NORTH, reward=1.5),
        ),
    )

    assert advice.action is Action.MOVE_NORTH
    assert advice.is_usable
    assert advice.reliability_reason == "usable"
    assert advice.reliability >= 0.35


def test_episodic_advisor_reports_danger_risk() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=evaluations(),
        episodes=(
            episode(
                Action.MOVE_EAST,
                reward=-10.0,
                event=EventType.HIT_DANGER,
            ),
            episode(
                Action.MOVE_EAST,
                reward=-9.0,
                event=EventType.HIT_DANGER,
            ),
            episode(Action.MOVE_NORTH, reward=1.0),
        ),
    )

    assert advice.action is Action.MOVE_NORTH
    assert advice.risk_hit_danger == 0.0
    assert advice.best_event is EventType.VISITED_EMPTY


def test_episodic_advisor_ignores_actions_not_currently_available() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=(ActionEvaluation(Action.MOVE_EAST, 10.0),),
        episodes=(
            episode(Action.MOVE_NORTH, reward=10.0),
            episode(Action.MOVE_EAST, reward=-1.0),
        ),
    )

    assert advice.action is Action.MOVE_EAST
    assert advice.expected_reward < 0.0


def test_rare_food_reward_is_dampened_until_repeated() -> None:
    advice = advise_from_episodes(
        agent=agent(),
        plan=plan(),
        evaluations=evaluations(),
        episodes=(
            episode(
                Action.MOVE_EAST,
                reward=15.0,
                event=EventType.ATE_FOOD,
            ),
            episode(Action.MOVE_NORTH, reward=1.0),
            episode(Action.MOVE_NORTH, reward=1.0),
            episode(Action.MOVE_NORTH, reward=1.0),
        ),
    )

    assert advice.has_advice
    assert advice.expected_reward < advice.raw_expected_reward
    assert "rare_reward_event_dampened" in advice.reliability_reason


def test_action_stats_use_only_top_similar_matches_per_action() -> None:
    query = build_episode_query(
        agent=agent(),
        plan=plan(),
        evaluations=(ActionEvaluation(Action.MOVE_EAST, 10.0),),
    )
    old_distant_episodes = tuple(
        episode(
            Action.MOVE_EAST,
            reward=-100.0,
            position_before=(10, 10),
            step=index,
        )
        for index in range(40)
    )
    recent_similar_episodes = tuple(
        episode(
            Action.MOVE_EAST,
            reward=2.0,
            position_before=(1, 1),
            step=100 + index,
        )
        for index in range(MAX_SIMILAR_EPISODES_PER_ACTION)
    )

    (stats,) = action_episode_stats(
        query=query,
        episodes=old_distant_episodes + recent_similar_episodes,
    )

    assert stats.match_count == MAX_SIMILAR_EPISODES_PER_ACTION
    assert stats.raw_expected_reward == 2.0


def test_weak_matches_below_similarity_floor_are_filtered() -> None:
    query = build_episode_query(
        agent=agent(),
        plan=plan(),
        evaluations=(ActionEvaluation(Action.MOVE_EAST, 10.0),),
    )
    strong_matches = tuple(
        episode(
            Action.MOVE_EAST,
            reward=2.0,
            position_before=(1, 1),
            step=100 + index,
        )
        for index in range(4)
    )
    weak_matches = tuple(
        episode(
            Action.MOVE_EAST,
            reward=-100.0,
            position_before=(17, 15),
            step=index,
        )
        for index in range(20)
    )

    (stats,) = action_episode_stats(
        query=query,
        episodes=weak_matches + strong_matches,
    )

    assert MIN_EPISODE_SIMILARITY_WEIGHT > 0.0
    assert MIN_RELATIVE_SIMILARITY_WEIGHT > 0.0
    assert stats.match_count == len(strong_matches)
    assert stats.raw_expected_reward == 2.0