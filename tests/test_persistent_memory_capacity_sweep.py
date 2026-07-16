from pathlib import Path

from experiments.run_persistent_memory_capacity_sweep import (
    CAPACITY_SWEEP_FIELDS,
    capacity_store_path,
    empty_capacity_summary,
)


def test_capacity_store_path_includes_cap() -> None:
    assert capacity_store_path(250) == Path(
        "artifacts/persistent_memory_capacity_250.jsonl"
    )


def test_empty_capacity_summary_matches_schema() -> None:
    row = empty_capacity_summary(
        max_prior_episodes=100,
        store_path=Path("artifacts/example.jsonl"),
        run_index=3,
        seed=23,
    )

    assert tuple(row) == CAPACITY_SWEEP_FIELDS
    assert row["max_prior_episodes"] == 100
    assert row["store_path"] == "artifacts\\example.jsonl" or row["store_path"] == "artifacts/example.jsonl"
    assert row["run_index"] == 3
    assert row["seed"] == 23
    assert row["prior_usable_reward_delta_is_defined"] is False
    assert row["termination_reason"] == "empty_history"
