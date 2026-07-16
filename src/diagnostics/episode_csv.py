import csv
from pathlib import Path
from typing import Any, Sequence

from src.diagnostics.csv_values import enum_csv, optional_csv
from src.memory.episode_trace import Episode, Position
from src.simulation.step_action_text import format_action


EPISODE_CSV_FIELDNAMES = (
    "step",
    "position_before_x",
    "position_before_y",
    "energy_before",
    "health_before",
    "curiosity_before",
    "goal_kind",
    "goal_target_x",
    "goal_target_y",
    "action",
    "reward",
    "event",
    "position_after_x",
    "position_after_y",
    "energy_after",
    "health_after",
    "curiosity_after",
    "network_action",
    "imagination_action",
    "choices_agree",
    "imagination_agrees",
)


def write_episodes_csv(
    episodes: Sequence[Episode],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=EPISODE_CSV_FIELDNAMES,
        )

        writer.writeheader()

        for episode in episodes:
            writer.writerow(episode_to_csv_row(episode))


def episode_to_csv_row(episode: Episode) -> dict[str, Any]:
    goal_target_x, goal_target_y = position_to_csv(
        episode.goal_target
    )

    return {
        "step": episode.step,
        "position_before_x": episode.position_before[0],
        "position_before_y": episode.position_before[1],
        "energy_before": episode.state_before[0],
        "health_before": episode.state_before[1],
        "curiosity_before": episode.state_before[2],
        "goal_kind": optional_csv(episode.goal_kind),
        "goal_target_x": goal_target_x,
        "goal_target_y": goal_target_y,
        "action": format_action(episode.action),
        "reward": episode.reward,
        "event": enum_csv(episode.event),
        "position_after_x": episode.position_after[0],
        "position_after_y": episode.position_after[1],
        "energy_after": episode.state_after[0],
        "health_after": episode.state_after[1],
        "curiosity_after": episode.state_after[2],
        "network_action": format_action(episode.network_action),
        "imagination_action": format_action(
            episode.imagination_action
        ),
        "choices_agree": episode.choices_agree,
        "imagination_agrees": episode.imagination_agrees,
    }


def position_to_csv(
    position: Position | None,
) -> tuple[int | str, int | str]:
    if position is None:
        return "", ""

    return position
