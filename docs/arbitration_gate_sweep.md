# Arbitration gate sweep

This branch adds a threshold sweep for conservative arbitration candidates.

It does **not** let episodic memory control Baby Vice.

The executed policy remains:

```text
symbolic rule policy executes
advisors are measured
arbitration candidates are measured
no advisor controls Baby Vice
```

## Why this follows arbitration readiness

The arbitration-readiness diagnostic answered a broad question:

```text
When usable advisor advice disagrees with the rule policy,
is the executed rule reward usually bad or good?
```

That showed useful rescue signal, but also some override risk. The next step is not to hand over control. The next step is to ask which stricter confidence/reliability/expected-reward thresholds keep the rescue signal while suppressing override risk.

## Gate rule under test

For each episodic channel, a candidate is counted when all conditions hold:

```text
advisor action exists
advisor advice is usable
advisor action disagrees with the rule action
advisor expected reward >= min_expected_reward
advisor confidence >= min_confidence
advisor reliability >= min_reliability
```

The sweep tries several threshold combinations.

Current grids:

```text
min_expected_reward: -1.0, 0.0, 0.5
min_confidence:       0.35, 0.45, 0.55
min_reliability:      0.45, 0.55, 0.65
```

## Output

Run:

```powershell
python -m experiments.run_arbitration_gate_sweep
```

It writes:

```text
artifacts/arbitration_gate_sweep.csv
artifacts/arbitration_gate_sweep_memory.jsonl
```

Important columns:

```text
channel
min_expected_reward
min_confidence
min_reliability
usable_disagreement_count
candidate_count
rescue_candidate_count
override_risk_count
candidate_capture_rate
rescue_candidate_rate
override_risk_rate
readiness_score
mean_rule_reward_when_candidate
mean_expected_reward_when_candidate
mean_confidence_when_candidate
mean_reliability_when_candidate
```

## Interpretation

A good gate has:

```text
candidate_count > 0
rescue_candidate_rate high
override_risk_rate low
readiness_score positive
mean_rule_reward_when_candidate negative
```

That means the gate mostly fires when the rule policy did poorly.

A bad gate has:

```text
override_risk_rate high
readiness_score near zero or negative
mean_rule_reward_when_candidate positive
```

That means the gate often fires when the rule policy was already doing well.

## Counterfactual boundary

This still does not prove the advisor's alternative action would have been better.

It only says:

```text
Under this threshold set,
the advisor disagreed during steps where the executed rule action had a bad/good/neutral reward.
```

Actual control requires a later gated shadow-arbitration experiment and then a controlled intervention experiment.
