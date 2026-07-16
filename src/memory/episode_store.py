from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.configs import SimulationConfig
from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.memory.episode_trace import Episode, Position, StateVector

STORE_VERSION = 1


@dataclass(frozen=True, slots=True)
class EpisodeStoreRecord:
    source_run_id: str
    source_seed: int
    world_signature: str
    episode: Episode


@dataclass(frozen=True, slots=True)
class LoadedEpisodeStore:
    records: tuple[EpisodeStoreRecord, ...]
    path: Path
    world_signature: str

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(record.episode for record in self.records)



def world_signature(config: SimulationConfig) -> str:
    world = config.world

    return ":".join(
        (
            "gridworld-v1",
            f"width={world.width}",
            f"height={world.height}",
            f"food={world.food_count}",
            f"danger={world.danger_count}",
            f"mystery={world.mystery_count}",
        )
    )



def load_episode_store(
    config: SimulationConfig,
) -> LoadedEpisodeStore:
    signature = world_signature(config)
    path = Path(config.episodic_memory.store_path)

    if not config.episodic_memory.enabled or not path.exists():
        return LoadedEpisodeStore(
            records=(),
            path=path,
            world_signature=signature,
        )

    records = tuple(
        load_episode_records(
            path=path,
            world_signature=signature,
            require_signature_match=(
                config.episodic_memory.require_world_signature_match
            ),
            limit=config.episodic_memory.max_prior_episodes,
        )
    )

    return LoadedEpisodeStore(
        records=records,
        path=path,
        world_signature=signature,
    )



def load_episode_records(
    path: Path,
    world_signature: str,
    require_signature_match: bool,
    limit: int,
) -> tuple[EpisodeStoreRecord, ...]:
    records: list[EpisodeStoreRecord] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            record = record_from_json(data)

            if (
                require_signature_match
                and record.world_signature != world_signature
            ):
                continue

            records.append(record)

    if len(records) > limit:
        records = records[-limit:]

    return tuple(records)



def save_episode_store(
    config: SimulationConfig,
    episodes: Sequence[Episode],
    source_run_id: str,
) -> Path | None:
    if (
        not config.episodic_memory.enabled
        or not config.episodic_memory.save_after_run
    ):
        return None

    path = Path(config.episodic_memory.store_path)
    signature = world_signature(config)

    existing = (
        list(
            load_episode_records(
                path=path,
                world_signature=signature,
                require_signature_match=False,
                limit=config.episodic_memory.max_store_episodes,
            )
        )
        if path.exists()
        else []
    )

    new_records = [
        EpisodeStoreRecord(
            source_run_id=source_run_id,
            source_seed=config.runtime.random_seed,
            world_signature=signature,
            episode=episode,
        )
        for episode in episodes
    ]

    all_records = existing + new_records

    if len(all_records) > config.episodic_memory.max_store_episodes:
        all_records = all_records[-config.episodic_memory.max_store_episodes:]

    write_episode_records(
        path=path,
        records=all_records,
    )

    return path



def write_episode_records(
    path: Path,
    records: Iterable[EpisodeStoreRecord],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record_to_json(record),
                    sort_keys=True,
                )
            )
            file.write("\n")



def record_to_json(
    record: EpisodeStoreRecord,
) -> dict[str, Any]:
    return {
        "version": STORE_VERSION,
        "source_run_id": record.source_run_id,
        "source_seed": record.source_seed,
        "world_signature": record.world_signature,
        "episode": episode_to_json(record.episode),
    }



def record_from_json(data: dict[str, Any]) -> EpisodeStoreRecord:
    version = int(data.get("version", 0))

    if version != STORE_VERSION:
        raise ValueError(
            f"Unsupported episodic memory record version: {version}"
        )

    return EpisodeStoreRecord(
        source_run_id=str(data["source_run_id"]),
        source_seed=int(data["source_seed"]),
        world_signature=str(data["world_signature"]),
        episode=episode_from_json(data["episode"]),
    )



def episode_to_json(episode: Episode) -> dict[str, Any]:
    return {
        "step": episode.step,
        "position_before": episode.position_before,
        "state_before": episode.state_before,
        "goal_kind": episode.goal_kind,
        "goal_target": episode.goal_target,
        "goal_id": episode.goal_id,
        "action": episode.action.name,
        "reward": episode.reward,
        "event": episode.event.value,
        "position_after": episode.position_after,
        "state_after": episode.state_after,
        "network_action": episode.network_action.name,
        "imagination_action": episode.imagination_action.name,
        "choices_agree": episode.choices_agree,
        "imagination_agrees": episode.imagination_agrees,
    }



def episode_from_json(data: dict[str, Any]) -> Episode:
    return Episode(
        step=int(data["step"]),
        position_before=position_from_json(data["position_before"]),
        state_before=state_vector_from_json(data["state_before"]),
        goal_kind=optional_text(data.get("goal_kind")),
        goal_target=optional_position_from_json(data.get("goal_target")),
        goal_id=optional_text(data.get("goal_id")),
        action=Action[str(data["action"])],
        reward=float(data["reward"]),
        event=EventType(str(data["event"])),
        position_after=position_from_json(data["position_after"]),
        state_after=state_vector_from_json(data["state_after"]),
        network_action=Action[str(data["network_action"])],
        imagination_action=Action[str(data["imagination_action"])],
        choices_agree=bool(data["choices_agree"]),
        imagination_agrees=bool(data["imagination_agrees"]),
    )



def position_from_json(value: object) -> Position:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Invalid position: {value!r}")

    return int(value[0]), int(value[1])



def optional_position_from_json(value: object) -> Position | None:
    if value is None:
        return None

    return position_from_json(value)



def state_vector_from_json(value: object) -> StateVector:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"Invalid state vector: {value!r}")

    return float(value[0]), float(value[1]), float(value[2])



def optional_text(value: object) -> str | None:
    if value is None:
        return None

    return str(value)
