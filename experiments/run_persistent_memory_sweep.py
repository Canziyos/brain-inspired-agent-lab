from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from src.configs import EpisodicMemoryConfig, SimulationConfig
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
STORE_PATH: Final[Path] = Path("artifacts") / "persistent_memory_sweep.jsonl"
OUTPUT_PATH: Final[Path] = Path("artifacts") / "persistent_memory_sweep.csv"

SWEEP_FIELDS: Final[tuple[str, ...]] = (
    "run_index",
    "seed",
    "prior_episode_count",
    "steps",
    "terminated",
    "truncated",
    "termination_reason",
    "total_reward",
    "mean_reward",
    "final_energy",
    "final_health",
    "seen_ratio",
    "visited_ratio",
    "unseen_cell_count",
    "semantic_goal_switches",
    "target_switches",
    "episodic_advice_rate",
    "episodic_usable_rate",
    "prior_advice_rate",
    "prior_usable_rate",
    "prior_rule_agreement",
    "prior_usable_rule_agreement",
    "prior_usable_imagination_agreement",
    "prior_advice_mean_reward",
    "prior_usable_mean_reward",
    "prior_weak_mean_reward",
    "prior_usable_reward_delta",
)


def main() -> None:
    reset_sweep_store(STORE_PATH)

    rows = run_sweep(DEFAULT_SEEDS)

    write_sweep_csv(
        rows=rows,
        output_path=OUTPUT_PATH,
    )

    print(f"Wrote persistent memory sweep to {OUTPUT_PATH}")
    print(f"Persistent memory sweep store: {STORE_PATH}")


def reset_sweep_store(path: Path) -> None:
    if path.exists():
        path.unlink()


def run_sweep(
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for index, seed in enumerate(seeds):
        history = run_simulation(
            make_persistent_config(seed)
        )
        rows.append(
            summarize_run(
                run_index=index,
                seed=seed,
                history=history,
            )
        )

    return rows


def make_persistent_config(seed: int) -> SimulationConfig:
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
) -> dict[str, object]:
    if not history:
        return empty_summary(
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

    return {
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
        "prior_usable_reward_delta": prior_usable_mean - prior_weak_mean,
    }


def empty_summary(
    run_index: int,
    seed: int,
) -> dict[str, object]:
    return {
        "run_index": run_index,
        "seed": seed,
        "prior_episode_count": 0,
        "steps": 0,
        "terminated": False,
        "truncated": False,
        "termination_reason": "empty_history",
        "total_reward": 0.0,
        "mean_reward": 0.0,
        "final_energy": 0.0,
        "final_health": 0.0,
        "seen_ratio": 0.0,
        "visited_ratio": 0.0,
        "unseen_cell_count": 0,
        "semantic_goal_switches": 0,
        "target_switches": 0,
        "episodic_advice_rate": 0.0,
        "episodic_usable_rate": 0.0,
        "prior_advice_rate": 0.0,
        "prior_usable_rate": 0.0,
        "prior_rule_agreement": 0.0,
        "prior_usable_rule_agreement": 0.0,
        "prior_usable_imagination_agreement": 0.0,
        "prior_advice_mean_reward": 0.0,
        "prior_usable_mean_reward": 0.0,
        "prior_weak_mean_reward": 0.0,
        "prior_usable_reward_delta": 0.0,
    }


def mean_reward(items: Sequence[StepMetrics]) -> float:
    if not items:
        return 0.0

    return sum(item.reward for item in items) / len(items)


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


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


if __name__ == "__main__":
    main()
