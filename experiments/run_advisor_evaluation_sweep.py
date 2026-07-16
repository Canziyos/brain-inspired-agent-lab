from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from src.configs import EpisodicMemoryConfig, SimulationConfig
from src.diagnostics.advisor_evaluation_csv import (
    ADVISOR_EVALUATION_FIELDS,
    summarize_advisor_evaluation,
)
from src.simulation.runner import run_simulation
from src.telemetry.metrics import StepMetrics

DEFAULT_SEEDS: Final[tuple[int, ...]] = (
    7,
    11,
    19,
    23,
    31,
    37,
    41,
    53,
    61,
    73,
)
STORE_PATH: Final[Path] = (
    Path("artifacts") / "advisor_evaluation_sweep_memory.jsonl"
)
OUTPUT_PATH: Final[Path] = (
    Path("artifacts") / "advisor_evaluation_sweep.csv"
)

SWEEP_FIELDS: Final[tuple[str, ...]] = (
    "run_index",
    "seed",
    *ADVISOR_EVALUATION_FIELDS,
)



def main() -> None:
    reset_store(STORE_PATH)
    rows = run_advisor_evaluation_sweep(DEFAULT_SEEDS)
    write_sweep_csv(
        rows=rows,
        output_path=OUTPUT_PATH,
    )

    print(f"Wrote advisor evaluation sweep to {OUTPUT_PATH}")
    print(f"Advisor evaluation memory store: {STORE_PATH}")



def reset_store(path: Path) -> None:
    if path.exists():
        path.unlink()



def run_advisor_evaluation_sweep(
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for run_index, seed in enumerate(seeds):
        history = run_simulation(
            make_config(seed)
        )
        rows.extend(
            summarize_run(
                run_index=run_index,
                seed=seed,
                history=history,
            )
        )

    return rows



def make_config(seed: int) -> SimulationConfig:
    base = SimulationConfig()

    return replace(
        base,
        runtime=replace(
            base.runtime,
            random_seed=seed,
            torch_seed=seed,
        ),
        output=replace(
            base.output,
            show_animation=False,
            show_plots=False,
            save_run_outputs=False,
        ),
        episodic_memory=EpisodicMemoryConfig(
            enabled=True,
            store_path=str(STORE_PATH),
            max_prior_episodes=5_000,
            max_store_episodes=20_000,
            require_world_signature_match=True,
            save_after_run=True,
        ),
    )



def summarize_run(
    run_index: int,
    seed: int,
    history: Sequence[StepMetrics],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_index": run_index,
            "seed": seed,
            **row,
        }
        for row in summarize_advisor_evaluation(history)
    )



def write_sweep_csv(
    rows: Sequence[dict[str, object]],
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
            fieldnames=SWEEP_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)



if __name__ == "__main__":
    main()
