# Persistent episodic memory

This branch adds persistent episodic memory across simulation runs while keeping Baby Vice in shadow mode.

The symbolic rule policy still chooses every executed action. Persistent memory only contributes diagnostics and advisor signals.

## What is persisted

After a run, the current run's passive episode trace can be appended to a JSONL store:

```text
artifacts/episodic_memory.jsonl
```

Each line stores:

- store record version
- source run id
- source seed
- world/config signature
- one episode

An episode contains:

```text
state before -> semantic goal -> action -> reward/event -> state after
```

including `goal_id`, action, event, network action, and imagination action.

## Safety gates

The persisted episodes are loaded only when `episodic_memory.enabled` is true.

The default config keeps persistence disabled so baseline runs and sweeps are not accidentally contaminated.

When enabled, loading can require a matching world/config signature. The signature currently includes:

```text
world width
world height
food count
danger count
mystery count
```

The random seed is stored as metadata but is not part of the signature. This lets a persistent-memory probe test whether previous runs can act as prior experience for later seeds under the same task family.

## Advisor behavior

At decision time the advisor now sees two histories:

```text
prior episodes from the persistent store
same-run episodes from the current trace
```

The existing episodic advice is computed from both histories together.

The branch also logs prior-run diagnostics separately:

- `episodic_prior_episode_count`
- `episodic_same_run_episode_count`
- `episodic_prior_action`
- `episodic_prior_is_usable`
- `episodic_prior_agrees_with_rule`
- `episodic_prior_agrees_with_imagination`

This lets us answer whether yesterday's notes are helping without letting them control behavior.

## Probe

Run:

```powershell
python -m experiments.run_persistent_memory_probe
```

The probe clears a dedicated store, runs seed 7 first, saves its episodes, then runs seed 11 with seed 7 loaded as prior memory.

It writes:

```text
artifacts/persistent_memory_probe.csv
artifacts/persistent_episodic_memory.jsonl
```

The important columns are:

- `prior_episode_count`
- `prior_advice_rate`
- `prior_usable_rate`
- `prior_rule_agreement`

## Still not policy control

Persistent memory is not an arbiter. It does not override rules, neural predictions, or imagination.

The next phase, after measuring persistence, can compare:

```text
same-run advice
prior-run advice
combined advice
rule policy
neural imagination
```

Only then should we consider policy arbitration.
