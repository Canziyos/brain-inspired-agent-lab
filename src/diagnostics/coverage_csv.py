import csv
from pathlib import Path
from typing import Any

from src.memory.working_memory import WorkingMemory


COVERAGE_CSV_FIELDNAMES = (
    "x",
    "y",
    "seen",
    "visited",
    "first_seen_step",
    "first_visited_step",
)


def write_coverage_csv(
    memory: WorkingMemory,
    width: int,
    height: int,
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
            fieldnames=COVERAGE_CSV_FIELDNAMES,
        )
        writer.writeheader()

        for row in coverage_rows(
            memory=memory,
            width=width,
            height=height,
        ):
            writer.writerow(row)


def coverage_rows(
    memory: WorkingMemory,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            position = (x, y)
            first_seen_step = memory.first_seen_step.get(position)
            first_visited_step = memory.first_visited_step.get(position)

            rows.append(
                {
                    "x": x,
                    "y": y,
                    "seen": first_seen_step is not None,
                    "visited": first_visited_step is not None,
                    "first_seen_step": optional_step(first_seen_step),
                    "first_visited_step": optional_step(
                        first_visited_step
                    ),
                }
            )

    return rows


def optional_step(step: int | None) -> int | str:
    if step is None:
        return ""

    return step
