from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from src.configs import SimulationConfig
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

OUTPUT_PATH: Final[Path] = Path("artifacts") / "episodic_advisor_sweep.csv"

SWEEP_FIELDS: Final[tuple[str, ...]] = (
    "seed",
    "steps",
    "terminated",
    "truncated",
    "termination_reason",
    "total_reward",
    "mean_reward",
    "final_energy",
    "final_health",
    "final_curiosity",
    "seen_ratio",
    "visited_ratio",
    "unseen_cell_count",
    "semantic_goal_switches",
    "target_switches",
    "episodic_advice_rate",
    "episodic_usable_rate",
    "episodic_rule_agreement",
    "episodic_imagination_agreement",
    "usable_episodic_rule_agreement",
    "usable_episodic_imagination_agreement",
    "usable_advice_mean_reward",
    "weak_advice_mean_reward",
    "mean_episodic_confidence",
    "mean_episodic_reliability",
    "mean_usable_reliability",
    "mean_danger_risk",
)


def main() -> None:
    rows = run_sweep(DEFAULT_SEEDS)
    write_sweep_csv(
        rows=rows,
        output_path=OUTPUT_PATH,
    )

    print(f"Wrote episodic advisor sweep to {OUTPUT_PATH}")


def run_sweep(
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for seed in seeds:
        history = run_simulation(
            make_sweep_config(seed)
        )
        rows.append(
            summarize_run(
                seed=seed,
                history=history,
            )
        )

    return rows


def make_sweep_config(seed: int) -> SimulationConfig:
    base = SimulationConfig()

    return replace(
        base,
        runtime=replace(
            base.runtime,
            random_seed=seed,
            torch_seed=seed,
            verbose=False,
        ),
        output=replace(
            base.output,
            show_animation=False,
            show_plots=False,
            save_run_outputs=False,
        ),
    )


def summarize_run(
    seed: int,
    history: Sequence[StepMetrics],
) -> dict[str, object]:
    if not history:
        return empty_row(seed)

    final = history[-1]
    total_reward = sum(item.reward for item in history)
    steps = len(history)
    advice_items = [item for item in history if item.episodic_has_advice]
    usable_items = [item for item in history if item.episodic_is_usable]
    weak_items = [
        item
        for item in history
        if item.episodic_has_advice and not item.episodic_is_usable
    ]

    return {
        "seed": seed,
        "steps": steps,
        "terminated": final.termination_reason in {"dead", "goals_complete"},
        "truncated": final.termination_reason == "time_limit",
        "termination_reason": final.termination_reason or "",
        "total_reward": total_reward,
        "mean_reward": total_reward / steps,
        "final_energy": final.energy,
        "final_health": final.health,
        "final_curiosity": final.curiosity,
        "seen_ratio": final.coverage_seen_ratio,
        "visited_ratio": final.coverage_visited_ratio,
        "unseen_cell_count": final.coverage_unseen_cell_count,
        "semantic_goal_switches": final.memory_goal_switch_count,
        "target_switches": final.memory_target_switch_count,
        "episodic_advice_rate": rate(len(advice_items), steps),
        "episodic_usable_rate": rate(len(usable_items), steps),
        "episodic_rule_agreement": agreement_rate(
            advice_items,
            attribute="episodic_agrees_with_rule",
        ),
        "episodic_imagination_agreement": agreement_rate(
            advice_items,
            attribute="episodic_agrees_with_imagination",
        ),
        "usable_episodic_rule_agreement": agreement_rate(
            usable_items,
            attribute="episodic_agrees_with_rule",
        ),
        "usable_episodic_imagination_agreement": agreement_rate(
            usable_items,
            attribute="episodic_agrees_with_imagination",
        ),
        "usable_advice_mean_reward": mean_reward(usable_items),
        "weak_advice_mean_reward": mean_reward(weak_items),
        "mean_episodic_confidence": mean(
            item.episodic_confidence for item in advice_items
        ),
        "mean_episodic_reliability": mean(
            item.episodic_reliability for item in advice_items
        ),
        "mean_usable_reliability": mean(
            item.episodic_reliability for item in usable_items
        ),
        "mean_danger_risk": mean(
            item.episodic_risk_hit_danger for item in advice_items
        ),
    }


def empty_row(seed: int) -> dict[str, object]:
    row: dict[str, object] = {
        field: ""
        for field in SWEEP_FIELDS
    }
    row["seed"] = seed
    row["steps"] = 0
    row["terminated"] = False
    row["truncated"] = False
    row["termination_reason"] = "no_history"
    return row


def agreement_rate(
    items: Sequence[StepMetrics],
    attribute: str,
) -> float:
    if not items:
        return 0.0

    return sum(
        bool(getattr(item, attribute))
        for item in items
    ) / len(items)


def mean_reward(items: Sequence[StepMetrics]) -> float:
    return mean(item.reward for item in items)


def mean(values: Sequence[float] | object) -> float:
    sequence = tuple(values)

    if not sequence:
        return 0.0

    return sum(sequence) / len(sequence)


def rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


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
