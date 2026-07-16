# Episodic top-K retrieval

This branch changes episodic retrieval from "use every matching memory" to "use the strongest local memories for each action."

The symbolic policy still chooses every action. Episodic retrieval remains diagnostic and advisory.

## Why

The persistent-memory capacity sweep showed a clear pattern:

```text
small prior-memory caps keep advice selective
large prior-memory caps make advice saturate
```

A global memory cap is useful as a diagnostic, but it is not the right cognitive mechanism. Baby Vice should be able to keep a large notebook while attending only to the most relevant entries for the current situation.

## Mechanism

For each candidate action, episodic retrieval now:

1. scores all candidate episodes by the existing similarity function
2. groups matches by action
3. sorts each action's matches by similarity weight
4. keeps only the top `MAX_SIMILAR_EPISODES_PER_ACTION`
5. computes reward, event, confidence, reliability, and danger risk from that bounded local set

The current cap is:

```text
MAX_SIMILAR_EPISODES_PER_ACTION = 32
```

This makes memory behave more like attention:

```text
large store -> local retrieval window -> calibrated advice
```

rather than:

```text
large store -> every similar-ish episode floods the advisor
```

## Expected effect

The next persistent-memory capacity sweep should show less saturation at high store sizes.

Good signs:

- `prior_usable_rate` no longer immediately goes to `1.0` for large stores
- `prior_usable_rule_agreement` stays meaningfully above raw prior agreement
- `prior_usable_reward_delta` remains positive when defined
- the small-cap advantage becomes less extreme

Bad signs:

- `prior_usable_rate` still saturates near `1.0`
- `prior_usable_rule_agreement` collapses toward raw agreement
- `prior_usable_reward_delta` becomes zero or negative

## Still shadow mode

This does not let memory control Baby Vice.

The action pipeline remains:

```text
symbolic rule policy executes
neural imagination is measured
episodic memory is measured
persistent memory is measured
```

Top-K retrieval only improves the advisor's evidence selection.