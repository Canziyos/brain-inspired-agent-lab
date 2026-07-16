# Arbitration readiness

This branch adds a shadow-mode analysis for deciding whether Baby Vice is close to safe policy arbitration.

The symbolic rule policy still executes every action. No advisor is allowed to steer the agent.

## Why this exists

Advisor evaluation showed that usable episodic agreement is strongly associated with better realized rule-policy reward. The next question is narrower:

```text
When a usable advisor disagrees with the rule policy,
is that disagreement usually a promising warning,
or is it an override risk?
```

This analysis does not answer the counterfactual question directly. It only examines the reward actually obtained by the rule policy on steps where an advisor agreed or disagreed.

## Channels

The sweep evaluates:

```text
same_run_episodic
prior_episodic
combined_episodic
neural_imagination
```

## Main concepts

A usable disagreement means:

```text
advisor has usable advice
advisor action != rule action
```

The disagreement is then categorized by the reward received by the executed rule action:

```text
reward < 0 -> rescue_candidate
reward > 0 -> override_risk
reward = 0 -> neutral_disagreement
```

Interpretation:

- `rescue_candidate`: the advisor disagreed when the executed rule action went badly.
- `override_risk`: the advisor disagreed when the executed rule action went well.
- `neutral_disagreement`: the executed rule action was neutral.

These names are diagnostic only. They do not prove the advisor's alternative action would have been better.

## Readiness score

For each channel:

```text
readiness_score = rescue_candidate_rate - override_risk_rate
```

A positive score means usable disagreements more often coincide with bad rule-policy outcomes than good rule-policy outcomes.

A negative score means the advisor often disagrees when the rule policy is doing well, so direct arbitration would be risky.

## Run

```powershell
python -m experiments.run_arbitration_readiness_sweep
```

Output:

```text
artifacts/arbitration_readiness_sweep.csv
artifacts/arbitration_readiness_memory.jsonl
```

## Next decision

If the combined episodic channel has a clearly positive readiness score and enough usable disagreement cases, the next branch can test a gated arbitration candidate in shadow mode.

That later branch should still avoid control at first. It should produce a proposed action and compare it against the executed rule action before allowing any policy changes.
