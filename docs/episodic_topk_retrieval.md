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

The first top-K attempt used a fairly generous cap and did not materially reduce saturation. The stricter version now combines a smaller per-action cap with an absolute and relative similarity floor.

## Mechanism

For each candidate action, episodic retrieval now:

1. scores all candidate episodes by the existing similarity function
2. groups matches by action
3. sorts each action's matches by similarity weight
4. removes weak local matches below the similarity floor
5. keeps only the strongest `MAX_SIMILAR_EPISODES_PER_ACTION`
6. computes reward, event, confidence, reliability, and danger risk from that bounded local set

The current retrieval limits are:

```text
MAX_SIMILAR_EPISODES_PER_ACTION = 8
MIN_EPISODE_SIMILARITY_WEIGHT = 0.08
MIN_RELATIVE_SIMILARITY_WEIGHT = 0.35
```

The relative floor means an episode must be at least 35% as similar as the strongest available match for the same action. The absolute floor prevents very weak matches from accumulating just because the memory store is large.

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

- `prior_usable_rate` drops for large stores compared with the loose top-K run
- `prior_usable_rule_agreement` stays meaningfully above raw prior agreement
- `prior_usable_reward_delta` remains positive when defined
- the small-cap advantage becomes less extreme

Bad signs:

- `prior_usable_rate` still saturates near `1.0`
- `prior_usable_rule_agreement` collapses toward raw agreement
- `prior_usable_reward_delta` becomes zero or negative

## Validation target

Run the capacity sweep again and compare the new CSV with the previous loose top-K output. The strict retrieval version should reduce the high-cap `prior_usable_rate` while keeping useful memories visibly better than weak memories.

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