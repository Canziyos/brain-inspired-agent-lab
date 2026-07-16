# Persistent memory capacity sweep

This experiment measures whether Baby Vice's persistent episodic advisor behaves differently when the prior-memory store is capped to different sizes.

Persistent memory remains in shadow mode. The symbolic rule policy still chooses every executed action.

## Why this sweep exists

The multi-seed persistent-memory sweep showed that prior memory is loaded and queried successfully. It also showed an important calibration issue: once the store becomes large, the prior advisor can mark nearly every prior suggestion as usable.

That is not automatically wrong, but it means we need to measure whether loading all prior episodes is better than loading a smaller, cleaner memory window.

This sweep compares multiple `max_prior_episodes` caps:

```text
100
250
500
1000
5000
```

Each cap gets its own independent JSONL store, so the cap comparisons do not contaminate one another.

## Run

```powershell
python -m experiments.run_persistent_memory_capacity_sweep
```

The script writes:

```text
artifacts/persistent_memory_capacity_sweep.csv
artifacts/persistent_memory_capacity_100.jsonl
artifacts/persistent_memory_capacity_250.jsonl
artifacts/persistent_memory_capacity_500.jsonl
artifacts/persistent_memory_capacity_1000.jsonl
artifacts/persistent_memory_capacity_5000.jsonl
```

## Main fields

The output includes all fields from the persistent-memory sweep plus cap-specific diagnostics:

```text
max_prior_episodes
prior_usable_count
prior_weak_count
prior_usable_reward_delta_is_defined
store_path
```

The most important comparisons are:

```text
prior_usable_rate
prior_usable_rule_agreement
prior_usable_reward_delta
prior_usable_reward_delta_is_defined
```

`prior_usable_reward_delta` is only meaningful when both usable and weak prior-memory advice exist in the same run. The boolean `prior_usable_reward_delta_is_defined` marks that case explicitly.

## Interpretation

Useful signs:

```text
prior_usable_reward_delta > 0
prior_usable_rule_agreement stays high
prior_usable_rate does not saturate immediately to 100%
```

Suspicious signs:

```text
prior_usable_rate = 100% for most later runs
prior_usable_rule_agreement collapses to raw prior agreement
prior_usable_reward_delta_is_defined = False for many runs
```

Those signs suggest the persistent store is too broad or the reliability gate is too permissive when enough prior episodes accumulate.

## Still not policy control

This experiment is still diagnostic-only. It does not let prior memory choose actions.

The next decision after this sweep is whether to add a memory-window strategy, a stricter prior-memory reliability gate, or source-diversity diagnostics before any policy arbitration.
