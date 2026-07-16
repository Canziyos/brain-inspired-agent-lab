from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from src.configs import EpisodicMemoryConfig, SimulationConfig
from src.simulation.runner import run_simulation
from src.telemetry.metrics import StepMetrics

STORE_PATH: Final[Path] = Path("artifacts") / "persistent_episodic_memory.jsonl"
OUTPUT_PATH: Final[Path] = Path("artifacts") / "persistent_memory_probe.csv"
DEFAULT_SEEDS: Final[tuple[int, ...]] = (7, 11)

PROBE_FIELDS: Final[tuple[str, ...]] = (
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
    "seen_ratio",
    "episodic_advice_rate",
    "episodic_usable_rate",
    "prior_advice_rate",
    "prior_usable_rate",
    "prior_rule_agreement",
)



def main() -> None:
    reset_probe_store(STORE_PATH)

    rows = run_probe(DEFAULT_SEEDS)

    write_probe_csv(
        rows=rows,
        output_path=OUTPUT_PATH,
    )

    print(f"Wrote persistent memory probe to {OUTPUT_PATH}")
    print(f"Persistent episodic memory store: {STORE_PATH}")



def reset_probe_store(path: Path) -> None:
    if path.exists():
        path.unlink()



def run_probe(seeds: Sequence[int]) -> list[dict[str, object]]:
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
            max_prior_episodes=2_000,
            max_store_episodes=10_000,
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
            "seen_ratio": 0.0,
            "episodic_advice_rate": 0.0,
            "episodic_usable_rate": 0.0,
            "prior_advice_rate": 0.0,
            "prior_usable_rate": 0.0,
            "prior_rule_agreement": 0.0,
        }

    final = history[-1]
    total_reward = sum(item.reward for item in history)
    advice_count = sum(item.episodic_has_advice for item in history)
    usable_count = sum(item.episodic_is_usable for item in history)
    prior_advice_count = sum(
        item.episodic_prior_action is not None
        for item in history
    )
    prior_usable_count = sum(
        item.episodic_prior_is_usable
        for item in history
    )
    prior_rule_agreement_count = sum(
        item.episodic_prior_agrees_with_rule
        for item in history
        if item.episodic_prior_action is not None
    )

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
        "seen_ratio": final.coverage_seen_ratio,
        "episodic_advice_rate": advice_count / len(history),
        "episodic_usable_rate": usable_count / len(history),
        "prior_advice_rate": prior_advice_count / len(history),
        "prior_usable_rate": prior_usable_count / len(history),
        "prior_rule_agreement": safe_rate(
            numerator=prior_rule_agreement_count,
            denominator=prior_advice_count,
        ),
    }



def write_probe_csv(
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
            fieldnames=PROBE_FIELDS,
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
