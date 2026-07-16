# Working memory and passive episodic trace

This branch adds a small cognitive memory layer without letting it overrule the symbolic policy.

## Working memory

Working memory tracks short-term context during a run:

- the current goal kind, target, and semantic goal id
- how long the current semantic goal has been active
- semantic goal switches versus raw target switches
- frontier target switches versus frontier semantic switches
- recent positions, actions, rewards, energies, and events
- a simple stuck counter
- last observed food, mystery, and danger positions

The planner receives a goal preference from working memory. If the current goal is still reachable, it receives a continuation bonus, and Baby Vice only switches away when another goal is clearly better.

## Semantic frontier identity

Food and mystery goals use exact target identity:

```text
food:3:4
mystery:10:2
```

Frontier goals use a cluster identity instead of an exact tile:

```text
frontier:0:5
```

This matters because frontier targets naturally move as exploration advances. A new frontier tile in the same frontier cluster is not a new intention; it is the same exploration mission moving along the boundary.

The diagnostics now separate:

```text
semantic_goal_switches = changes in intended goal identity
target_switches        = changes in exact coordinate target
```

So Baby Vice is not punished for a frontier boundary sliding one tile like an idiot cloud.

## Frontier reachability and clusters

The frontier planner now separates:

```text
frontier_count
reachable_frontier_count
unreachable_frontier_count
frontier_cluster_count
reachable_frontier_cluster_count
current_frontier_cluster_id
```

A raw frontier means a known traversable cell touches unknown space. A reachable frontier means Baby Vice can actually path to it through known traversable space.

This is important because a run can finish all reachable goals while still having a few raw frontier cells near unknown space. Those leftover cells should not be confused with useful exploration opportunities.

## Perception coverage diagnostics

Baby Vice currently senses only adjacent cells. Coverage diagnostics separate three different concepts:

```text
visited = cells Baby Vice physically stepped on
known   = cells stored in the agent's symbolic map
seen    = known cells plus physically visited cells
```

Working memory now tracks:

- total world cells
- seen cells and seen ratio
- visited cells and visited ratio
- unseen cell count
- raw frontier count per step
- reachable and unreachable frontier counts
- raw and reachable frontier cluster counts
- current frontier cluster id
- newly seen cells per step
- newly visited cells per step
- first step each cell was seen
- first step each cell was visited

This lets us measure whether Baby Vice really investigated the world or simply finished the reachable goals.

## Passive episodic trace

The episodic trace records one row per step:

```text
state before -> goal -> action -> event/reward -> state after
```

Episodes also record `goal_id`, so later episodic retrieval can compare semantic goal identities rather than only coordinate targets.

This is shadow-mode memory only. It does not retrieve episodes and does not challenge the symbolic policy yet.

Run outputs now include:

- `steps.csv`, extended with working-memory and coverage diagnostics
- `episodes.csv`, containing the passive episodic trace
- `coverage.csv`, containing one row per world cell with first-seen and first-visited steps

## Intended next step

After this branch is measured, episodic retrieval can be added in shadow mode as an advisor. It should not control policy until its advice is measurable and stable.

A separate later branch can expose `vision_radius` as an experiment setting and compare radius 1, 2, and 3 using these coverage diagnostics.
