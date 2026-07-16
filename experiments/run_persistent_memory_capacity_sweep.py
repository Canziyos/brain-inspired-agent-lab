from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from src.configs import EpisodicMemoryConfig, SimulationConfig
from src.simulation.runner import run_simulation
from src.telemetry.metrics import StepMetrics

from experiments.run_persistent_memory_sweep import (
    DEFAULT_SEEDS,
    SWEEP_FIELDS,
    mean_reward,
    reset_sweep_store,
    safe_rate,
)

DEFAULT_CAPS: Final[tuple[int, ...]] = (
    100,
    250,
    500,
    1_000,
    5_000,
)
OUTPUT_PATH: Final[Path] = (
    Path("artifacts") / "persistent_memory_capacity_sweep.csv"
)

CAPACITY_SWEEP_FIELDS: Final[tuple[str, ...]] = (
    "max_prior_episodes",
    "prior_usable_count",
    "prior_weak_count",
    "prior_usable_reward_delta_is_defined",
    "store_path",
    *SWEEP_FIELDS,
)


def main() -> None:
    rows = run_capacity_sweep(
        seeds=DEFAULT_SEEDS,
        caps=DEFAULT_CAPS,
    )

    write_capacity_sweep_csv(
        rows=rows,
        output_path=OUTPUT_PATH,
    )

    print(f"Wrote persistent memory capacity sweep to {OUTPUT_PATH}")


def run_capacity_sweep(
    seeds: Sequence[int],
    caps: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for cap in caps:
        store_path = capacity_store_path(cap)
        reset_sweep_store(store_path)

        for index, seed in enumerate(seeds):
            history = run_simulation(
                make_capacity_config(
                    seed=seed,
                    max_prior_episodes=cap,
                    store_path=store_path,
                )
            )
            rows.append(
                summarize_capacity_run(
                    max_prior_episodes=cap,
                    store_path=store_path,
                    run_index=index,
                    seed=seed,
                    history=history,
                )
            )

    return rows


def capacity_store_path(max_prior_episodes: int) -> Path:
    return (
        Path("artifacts")
        / f"persistent_memory_capacity_{max_prior_episodes}.jsonl"
    )


def make_capacity_config(
    seed: int,
    max_prior_episodes: int,
    store_path: Path,
) -> SimulationConfig:
    base = SimulationConfig()

    return replace(
        base,
        runtime=replace(
            base.runtime,
            random_seed=seed,
            torch_seed=seed,
        ),
        episodic_memory=EpisodicMemoryConfig(
            enabled=True,
            store_path=str(store_path),
            max_prior_episodes=max_prior_episodes,
            max_store_episodes=20_000,
            require_world_signature_match=True,
            save_after_run=True,
        ),
    )


def summarize_capacity_run(
    max_prior_episodes: int,
    store_path: Path,
    run_index: int,
    seed: int,
    history: Sequence[StepMetrics],
) -> dict[str, object]:
    if not history:
        return empty_capacity_summary(
            max_prior_episodes=max_prior_episodes,
            store_path=store_path,
            run_index=run_index,
            seed=seed,
        )

    final = history[-1]
    total_reward = sum(item.reward for item in history)

    advice_count = sum(item.episodic_has_advice for item in history)
    usable_count = sum(item.episodic_is_usable for item in history)

    prior_advice_items = tuple(
        item
        for item in history
        if item.episodic_prior_action is not None
    )
    prior_usable_items = tuple(
        item
        for item in history
        if item.episodic_prior_is_usable
    )
    prior_weak_items = tuple(
        item
        for item in prior_advice_items
        if not item.episodic_prior_is_usable
    )

    prior_usable_mean = mean_reward(prior_usable_items)
    prior_weak_mean = mean_reward(prior_weak_items)
    delta_is_defined = bool(prior_usable_items and prior_weak_items)

    return {
        "max_prior_episodes": max_prior_episodes,
        "prior_usable_count": len(prior_usable_items),
        "prior_weak_count": len(prior_weak_items),
        "prior_usable_reward_delta_is_defined": delta_is_defined,
        "store_path": str(store_path),
        "run_index": run_index,
        "seed": seed,
        "prior_episode_count": final.episodic_prior_episode_count,
        "steps": len(history),
        "terminated": final.termination_reason != "time_limit",
        "truncated": final.termination_reason == "time_limit",
        "termination_reason": final.termination_reason or "",
        "total_reward": total_reward,
        "mean_reward": total_reward / len(history),
        "final_energy": final.energy,
        "final_health": final.health,
        "seen_ratio": final.coverage_seen_ratio,
        "visited_ratio": final.coverage_visited_ratio,
        "unseen_cell_count": final.coverage_unseen_cell_count,
        "semantic_goal_switches": final.memory_goal_switch_count,
        "target_switches": final.memory_target_switch_count,
        "episodic_advice_rate": advice_count / len(history),
        "episodic_usable_rate": usable_count / len(history),
        "prior_advice_rate": len(prior_advice_items) / len(history),
        "prior_usable_rate": len(prior_usable_items) / len(history),
        "prior_rule_agreement": safe_rate(
            numerator=sum(
                item.episodic_prior_agrees_with_rule
                for item in prior_advice_items
            ),
            denominator=len(prior_advice_items),
        ),
        "prior_usable_rule_agreement": safe_rate(
            numerator=sum(
                item.episodic_prior_agrees_with_rule
                for item in prior_usable_items
            ),
            denominator=len(prior_usable_items),
        ),
        "prior_usable_imagination_agreement": safe_rate(
            numerator=sum(
                item.episodic_prior_agrees_with_imagination
                for item in prior_usable_items
            ),
            denominator=len(prior_usable_items),
        ),
        "prior_advice_mean_reward": mean_reward(prior_advice_items),
        "prior_usable_mean_reward": prior_usable_mean,
        "prior_weak_mean_reward": prior_weak_mean,
        "prior_usable_reward_delta": (
            prior_usable_mean - prior_weak_mean
            if delta_is_defined
            else 0.0
        ),
    }


def empty_capacity_summary(
    max_prior_episodes: int,
    store_path: Path,
    run_index: int,
    seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        field: 0
        for field in CAPACITY_SWEEP_FIELDS
    }
    row.update(
        {
            "max_prior_episodes": max_prior_episodes,
            "prior_usable_count": 0,
            "prior_weak_count": 0,
            "prior_usable_reward_delta_is_defined": False,
            "store_path": str(store_path),
            "run_index": run_index,
            "seed": seed,
            "terminated": False,
            "truncated": False,
            "termination_reason": "empty_history",
        }
    )
    return row


def write_capacity_sweep_csv(
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
            fieldnames=CAPACITY_SWEEP_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
