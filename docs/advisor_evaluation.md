# Advisor evaluation

This branch adds a shadow-mode evaluator for comparing Baby Vice's decision sources without giving any of them control.

The executed action is still chosen by the symbolic rule policy.

## Why

After persistent memory and top-K retrieval, we need a clean way to compare advisor channels:

```text
rule policy
same-run episodic advice
prior-run episodic advice
combined episodic advice
neural imagination
```

The evaluator does not estimate full counterfactual rewards. It only measures realized rewards on steps where an advisor agreed or disagreed with the executed rule action.

That distinction matters:

```text
agreement reward = reward received when advisor action matched the executed rule action
disagreement reward = reward received by the rule action when advisor disagreed
```

So disagreement reward is not the reward the advisor would have received. It is a safety-first diagnostic before policy arbitration.

## Run artifact

Normal saved runs now write:

```text
advisor_evaluation.csv
```

beside:

```text
steps.csv
episodes.csv
coverage.csv
```

The advisor evaluation file is long-form. Each row is one channel.

Important columns:

- `channel`
- `advice_count`
- `usable_count`
- `agreement_rate`
- `usable_agreement_rate`
- `mean_rule_reward_when_advice_agreed`
- `mean_rule_reward_when_advice_disagreed`
- `mean_rule_reward_when_usable_agreed`
- `mean_rule_reward_when_usable_disagreed`

## Sweep

Run:

```powershell
python -m experiments.run_advisor_evaluation_sweep
```

It runs the usual ten-seed persistent-memory sequence and writes:

```text
artifacts/advisor_evaluation_sweep.csv
artifacts/advisor_evaluation_sweep_memory.jsonl
```

The sweep output adds `run_index` and `seed` before the advisor-evaluation columns.

## Interpretation

Good signals:

- usable advice agrees with the rule policy more often than raw advice
- usable-agreed steps have higher realized rule reward than usable-disagreed steps
- prior memory remains selective after top-K retrieval
- combined episodic advice is not simply identical to prior or same-run advice

Bad signals:

- usable advice saturates near every step
- usable disagreement reward is strongly higher than usable agreement reward
- prior memory overwhelms same-run evidence too early

## Still shadow mode

This branch does not implement arbitration.

The purpose is to build the scoreboard needed before arbitration:

```text
who tends to agree with successful actions?
who disagrees in useful places?
who becomes overconfident?
```

Only after this evaluator is stable should Baby Vice be allowed a supervised arbitration layer.
