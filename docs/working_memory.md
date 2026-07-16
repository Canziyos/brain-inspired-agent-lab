# Working memory and passive episodic trace

This branch adds a small cognitive memory layer without letting it overrule the symbolic policy.

## Working memory

Working memory tracks short-term context during a run:

- the current goal kind and target
- how long the current goal has been active
- how many goal switches happened
- recent positions, actions, rewards, energies, and events
- a simple stuck counter
- last observed food, mystery, and danger positions

The planner receives a goal preference from working memory. If the current goal is still reachable, it receives a continuation bonus, and Baby Vice only switches away when another goal is clearly better.

## Passive episodic trace

The episodic trace records one row per step:

```text
state before -> goal -> action -> event/reward -> state after
```

This is shadow-mode memory only. It does not retrieve episodes and does not challenge the symbolic policy yet.

Run outputs now include:

- `steps.csv`, extended with working-memory diagnostics
- `episodes.csv`, containing the passive episodic trace

## Intended next step

After this branch is measured, episodic retrieval can be added in shadow mode as an advisor. It should not control policy until its advice is measurable and stable.
