from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.memory.episode_retrieval import (
    NO_EPISODIC_ADVICE,
    advise_from_episodes,
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
) -> Episode:
    return Episode(
        step=0,
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


def test_no_episodes_produces_no_advice() -> None:
    agent = Agent(x=1, y=1, energy=50.0, curiosity=20.0)

    advice = advise_from_episodes(
        agent=agent,
        plan=plan(),
        evaluations=evaluations(),
        episodes=(),
    )

    assert advice == NO_EPISODIC_ADVICE
    assert not advice.has_advice


def test_episodic_advisor_prefers_action_with_better_memory() -> None:
    agent = Agent(x=1, y=1, energy=50.0, curiosity=20.0)

    advice = advise_from_episodes(
        agent=agent,
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
    assert advice.expected_reward > 0.0
    assert advice.match_count == 2
    assert advice.confidence > 0.0
    assert advice.has_advice


def test_episodic_advisor_reports_danger_risk() -> None:
    agent = Agent(x=1, y=1, energy=50.0, curiosity=20.0)

    advice = advise_from_episodes(
        agent=agent,
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
    agent = Agent(x=1, y=1, energy=50.0, curiosity=20.0)

    advice = advise_from_episodes(
        agent=agent,
        plan=plan(),
        evaluations=(ActionEvaluation(Action.MOVE_EAST, 10.0),),
        episodes=(
            episode(Action.MOVE_NORTH, reward=10.0),
            episode(Action.MOVE_EAST, reward=-1.0),
        ),
    )

    assert advice.action is Action.MOVE_EAST
    assert advice.expected_reward < 0.0
