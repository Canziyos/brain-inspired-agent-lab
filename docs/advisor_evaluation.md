# Advisor evaluation diagnostics

This branch adds a shadow-mode scoreboard for Baby Vice's advisors.

The symbolic rule policy still executes every action. The evaluator only asks which advisory channels agreed with the executed action and what realized rule-policy rewards were observed on those agreement/disagreement subsets.

## Channels

The evaluator writes one row per channel:

```text
rule_policy
same_run_episodic
prior_episodic
combined_episodic
neural_imagination
```

The episodic channels mean:

- `same_run_episodic`: advice from episodes recorded earlier in the current run
- `prior_episodic`: advice from persistent memory loaded before the current run
- `combined_episodic`: advice from prior memory plus current-run memory

## Output files

For saved runs, the run directory now includes:

```text
advisor_evaluation.csv
```

beside:

```text
steps.csv
episodes.csv
coverage.csv
```

The experiment command:

```powershell
python -m experiments.run_advisor_evaluation_sweep
```

writes:

```text
artifacts/advisor_evaluation_sweep.csv
artifacts/advisor_evaluation_sweep_memory.jsonl
```

## Metrics

Each channel row reports:

```text
advice_count
usable_count
agreement_count
usable_agreement_count
advice_rate
usable_rate
agreement_rate
usable_agreement_rate
mean_rule_reward_all_steps
mean_rule_reward_when_advice_agreed
mean_rule_reward_when_advice_disagreed
mean_rule_reward_when_usable_agreed
mean_rule_reward_when_usable_disagreed
mean_expected_reward
mean_confidence
mean_reliability
```

The `mean_expected_reward`, `mean_confidence`, and `mean_reliability` columns are populated for each advisory source that has those values:

- same-run episodic advice
- prior-run episodic advice
- combined episodic advice
- neural imagination expected reward

Rule policy has no advisory confidence/reliability, so those fields stay zero for that row.

## Important interpretation limit

The reward columns are not counterfactual rewards.

They are realized rewards from the symbolic rule policy:

```text
rule policy executed action
advisor agreed or disagreed
record realized rule reward
```

So this evaluator can support statements such as:

```text
When prior memory agreed with the executed rule action, those steps had higher realized rewards.
```

It cannot yet support:

```text
If prior memory had controlled Baby Vice, total return would have improved.
```

That requires an arbitration/control experiment later.

## Why this step exists

Top-K episodic retrieval reduced prior-memory saturation. Before we allow any advisor to influence behavior, we need a diagnostic scoreboard that reveals:

- whether same-run memory agrees with successful rule actions
- whether prior memory agrees with successful rule actions
- whether combined memory is better than either memory source alone
- whether neural imagination is aligned with rule behavior
- whether advisor disagreements identify possible missed opportunities

## Desired signs

Good signals:

```text
usable agreement subsets have better realized rule rewards than disagreement subsets
prior and combined memory keep positive reward separation
same-run memory becomes reliable after enough current-run evidence accumulates
```

Warning signs:

```text
usable disagreements are consistently better than usable agreements
combined memory becomes worse than prior-only or same-run-only
advisor confidence/reliability rise while reward separation collapses
```

## Still shadow mode

This branch does not introduce policy arbitration.

The action pipeline remains:

```text
symbolic rule policy executes
same-run memory is measured
prior memory is measured
combined memory is measured
neural imagination is measured
```

Baby Vice receives a scoreboard, not a steering wheel.
