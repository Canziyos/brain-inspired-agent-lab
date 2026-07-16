# Episodic advisor shadow mode

This branch turns the passive episode trace into a read-only advisor.

The advisor does **not** choose actions. The symbolic rule policy remains the controller.

## What the advisor does

At each step, the advisor receives:

- Baby Vice's current position and internal state
- the selected semantic goal, including `goal_id`
- the currently available action evaluations
- previous episodes from the same run

It searches previous episodes for similar situations and scores each currently available action by weighted historical reward.

Similarity uses:

- semantic goal identity (`goal_id`) when available
- goal kind when exact identity does not match
- spatial distance between current position and episode position
- energy, health, and curiosity distance

## Calibration

The advisor now separates raw advice from usable advice.

```text
raw advice    = memory has an opinion
usable advice = memory has enough evidence
```

Calibration uses:

- minimum match count
- minimum confidence
- minimum reliability
- minimum similarity weight
- danger-risk gating
- reward shrinkage toward zero when evidence is weak
- dampening for rare high-reward events such as one lucky food pickup

This prevents early episodic memory from treating one lucky reward as universal wisdom.

## What gets logged

`steps.csv` now includes:

- `episodic_action`
- `episodic_raw_expected_reward`
- `episodic_expected_reward`
- `episodic_confidence`
- `episodic_reliability`
- `episodic_match_count`
- `episodic_best_event`
- `episodic_risk_hit_danger`
- `episodic_has_advice`
- `episodic_is_usable`
- `episodic_agrees_with_rule`
- `episodic_agrees_with_imagination`
- `episodic_reliability_reason`
- `episodic_rationale`

The run log summarizes:

- `episodic_advice_rate`
- `episodic_usable_rate`
- `episodic_rule_agreement`
- `episodic_imagination_agreement`
- `usable_episodic_rule_agreement`

## Multi-seed sweep

Run:

```powershell
python -m experiments.run_episodic_advisor_sweep
```

This writes:

```text
artifacts/episodic_advisor_sweep.csv
```

The sweep runs a fixed list of seeds and summarizes advisor rates, usable-advice rates, reward, coverage, goal switching, reliability, and danger-risk signals.

## Why shadow mode

Episodes are local and early memory can be noisy. If an unlucky action caused pain once, the agent should not immediately generalize that the action is cursed.

So for now, episodic memory only advises. Later we can add a policy arbiter that compares:

```text
rule policy
neural imagination
episodic advisor
```

and decides when memory is trustworthy enough to influence behavior.
