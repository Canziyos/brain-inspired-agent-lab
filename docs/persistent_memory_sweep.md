# Persistent-memory multi-seed sweep

This experiment measures whether prior-run episodic memory remains useful across a sequence of different random seeds.

Persistent memory is still shadow-mode only. The symbolic rule policy continues to choose every executed action.

## Run

```powershell
python -m experiments.run_persistent_memory_sweep
```

The script clears a dedicated sweep store, then runs these seeds in order:

```text
7, 11, 19, 23, 31, 37, 41, 53, 61, 73
```

Each run loads all compatible prior episodes from earlier sweep runs, executes the current seed, then appends the current run's episodes to the store.

## Outputs

```text
artifacts/persistent_memory_sweep.csv
artifacts/persistent_memory_sweep.jsonl
```

The JSONL file is the persistent episode store for this sweep.

The CSV file records one row per seed.

## Key columns

Core run outcomes:

```text
steps
total_reward
mean_reward
final_energy
seen_ratio
visited_ratio
unseen_cell_count
semantic_goal_switches
target_switches
```

Combined episodic advisor diagnostics:

```text
episodic_advice_rate
episodic_usable_rate
```

Prior-run memory diagnostics:

```text
prior_episode_count
prior_advice_rate
prior_usable_rate
prior_rule_agreement
prior_usable_rule_agreement
prior_usable_imagination_agreement
prior_advice_mean_reward
prior_usable_mean_reward
prior_weak_mean_reward
prior_usable_reward_delta
```

## Interpretation

The most important quality signal is:

```text
prior_usable_reward_delta = prior_usable_mean_reward - prior_weak_mean_reward
```

A positive value means the reliability gates are separating better prior memories from weak prior memories.

Because memory is not controlling the policy yet, this sweep should not be read as proof that persistent memory improves behavior. It measures whether persistent memory produces reliable advice signals across different worlds in the same task family.

## Next decision after this sweep

If prior usable memory stays positive across many seeds, the next safe step is policy arbitration in shadow mode:

```text
rule policy action
combined episodic action
prior-only episodic action
imagination action
```

Only after those shadows are stable should Baby Vice receive any limited steering authority.
