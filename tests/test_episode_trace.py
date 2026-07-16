from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.memory.episode_trace import (
    EpisodicTrace,
    build_episode,
)
from src.planning.goal_planner import GoalKind, GoalPlan


def test_episodic_trace_records_passive_episode() -> None:
    trace = EpisodicTrace()
    plan = GoalPlan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        path=((1, 0), (2, 0)),
    )

    episode = build_episode(
        step=3,
        position_before=(1, 0),
        state_before=(40.0, 100.0, 20.0),
        plan=plan,
        action=Action.MOVE_EAST,
        reward=2.0,
        event=EventType.DISCOVERED_MYSTERY,
        position_after=(2, 0),
        state_after=(38.0, 100.0, 15.0),
        network_action=Action.MOVE_NORTH,
        imagination_action=Action.MOVE_EAST,
        choices_agree=False,
        imagination_agrees=True,
    )

    trace.record(episode)

    assert len(trace) == 1
    assert trace.episodes[0].goal_kind == "mystery"
    assert trace.episodes[0].goal_target == (2, 0)
    assert trace.episodes[0].action is Action.MOVE_EAST
    assert trace.episodes[0].event is EventType.DISCOVERED_MYSTERY


def test_episode_without_goal_is_allowed() -> None:
    episode = build_episode(
        step=0,
        position_before=(0, 0),
        state_before=(70.0, 100.0, 50.0),
        plan=None,
        action=Action.REST,
        reward=-1.0,
        event=EventType.RESTED,
        position_after=(0, 0),
        state_after=(75.0, 100.0, 50.0),
        network_action=Action.REST,
        imagination_action=Action.REST,
        choices_agree=True,
        imagination_agrees=True,
    )

    assert episode.goal_kind is None
    assert episode.goal_target is None
