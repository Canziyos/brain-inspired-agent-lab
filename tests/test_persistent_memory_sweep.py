from experiments.run_persistent_memory_sweep import (
    SWEEP_FIELDS,
    empty_summary,
    safe_rate,
)


def test_empty_summary_contains_all_sweep_fields() -> None:
    row = empty_summary(
        run_index=3,
        seed=41,
    )

    assert tuple(row) == SWEEP_FIELDS
    assert row["run_index"] == 3
    assert row["seed"] == 41
    assert row["termination_reason"] == "empty_history"
    assert row["prior_usable_reward_delta"] == 0.0


def test_safe_rate_handles_empty_denominator() -> None:
    assert safe_rate(7, 0) == 0.0
    assert safe_rate(3, 6) == 0.5
