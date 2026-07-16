from dataclasses import replace

from src.configs import EpisodicMemoryConfig, SimulationConfig
from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.memory.episode_store import (
    EpisodeStoreRecord,
    load_episode_records,
    save_episode_store,
    world_signature,
    write_episode_records,
)
from src.memory.episode_trace import Episode



def make_episode(
    step: int,
    action: Action = Action.MOVE_EAST,
) -> Episode:
    return Episode(
        step=step,
        position_before=(1, 2),
        state_before=(50.0, 100.0, 20.0),
        goal_kind="frontier",
        goal_target=(2, 2),
        goal_id="frontier:1:2",
        action=action,
        reward=1.5,
        event=EventType.VISITED_EMPTY,
        position_after=(2, 2),
        state_after=(48.0, 100.0, 20.0),
        network_action=Action.REST,
        imagination_action=action,
        choices_agree=False,
        imagination_agrees=True,
    )



def config_with_store(path) -> SimulationConfig:
    base = SimulationConfig()

    return replace(
        base,
        episodic_memory=EpisodicMemoryConfig(
            enabled=True,
            store_path=str(path),
            max_prior_episodes=10,
            max_store_episodes=10,
        ),
    )



def test_episode_store_round_trips_episodes(tmp_path) -> None:
    store_path = tmp_path / "memory.jsonl"
    config = config_with_store(store_path)

    saved_path = save_episode_store(
        config=config,
        episodes=(make_episode(0), make_episode(1, Action.REST)),
        source_run_id="run-a",
    )

    assert saved_path == store_path

    records = load_episode_records(
        path=store_path,
        world_signature=world_signature(config),
        require_signature_match=True,
        limit=10,
    )

    assert len(records) == 2
    assert records[0].source_run_id == "run-a"
    assert records[0].source_seed == config.runtime.random_seed
    assert records[0].episode.action is Action.MOVE_EAST
    assert records[1].episode.action is Action.REST



def test_episode_store_filters_world_signature(tmp_path) -> None:
    store_path = tmp_path / "memory.jsonl"
    config = config_with_store(store_path)
    other_signature = "gridworld-v1:width=99:height=99"

    write_episode_records(
        path=store_path,
        records=(
            EpisodeStoreRecord(
                source_run_id="wrong-world",
                source_seed=1,
                world_signature=other_signature,
                episode=make_episode(0),
            ),
            EpisodeStoreRecord(
                source_run_id="right-world",
                source_seed=2,
                world_signature=world_signature(config),
                episode=make_episode(1),
            ),
        ),
    )

    records = load_episode_records(
        path=store_path,
        world_signature=world_signature(config),
        require_signature_match=True,
        limit=10,
    )

    assert [record.source_run_id for record in records] == [
        "right-world"
    ]



def test_episode_store_caps_loaded_records(tmp_path) -> None:
    store_path = tmp_path / "memory.jsonl"
    config = config_with_store(store_path)

    save_episode_store(
        config=config,
        episodes=tuple(make_episode(step) for step in range(5)),
        source_run_id="run-a",
    )

    records = load_episode_records(
        path=store_path,
        world_signature=world_signature(config),
        require_signature_match=True,
        limit=2,
    )

    assert [record.episode.step for record in records] == [3, 4]
