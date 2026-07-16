from src.core.actions import Action
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import CellType
from src.planning.frontier_planner import (
    find_frontier_clusters,
    find_frontiers,
    find_reachable_frontiers,
    frontier_cluster_for_position,
)
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
)
from src.policies.rule_policy import (
    choose_action,
    evaluate_actions,
)



def create_corridor_agent() -> Agent:
    agent = Agent(
        x=4,
        y=1,
        energy=80.0,
    )

    agent.visited.update(
        {
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        }
    )

    agent.known_cells.update(
        {
            (1, 1): CellType.EMPTY,
            (2, 1): CellType.EMPTY,
            (3, 1): CellType.EMPTY,
            (4, 1): CellType.EMPTY,

            (1, 0): CellType.DANGER,
            (1, 2): CellType.DANGER,
            (2, 0): CellType.DANGER,
            (2, 2): CellType.DANGER,
            (3, 0): CellType.DANGER,
            (3, 2): CellType.DANGER,
            (4, 0): CellType.DANGER,
            (4, 2): CellType.DANGER,
        }
    )

    return agent



def test_policy_prefers_frontier_travel_over_revisit() -> None:
    agent = create_corridor_agent()

    plan = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 1),
        path=(
            (4, 1),
            (3, 1),
            (2, 1),
            (1, 1),
        ),
    )

    evaluations = evaluate_actions(
        agent,
        observations=[
            Observation(3, 1, CellType.EMPTY),
            Observation(4, 0, CellType.DANGER),
            Observation(4, 2, CellType.DANGER),
        ],
        plan=plan,
    )

    chosen = choose_action(evaluations)

    assert chosen.action is Action.MOVE_WEST
    assert (
        chosen.rationale
        == "follow frontier goal toward (1, 1)"
    )



def test_reachable_frontiers_are_subset_of_raw_frontiers() -> None:
    agent = Agent(x=0, y=0)
    agent.remember_cell((1, 0), CellType.EMPTY)
    agent.remember_cell((0, 1), CellType.EMPTY)

    raw_frontiers = find_frontiers(
        agent=agent,
        width=3,
        height=3,
    )
    reachable_frontiers = find_reachable_frontiers(
        agent=agent,
        width=3,
        height=3,
    )

    assert reachable_frontiers
    assert reachable_frontiers <= raw_frontiers



def test_frontier_clusters_have_stable_anchor_ids() -> None:
    agent = Agent(x=0, y=0)
    agent.remember_cell((1, 0), CellType.EMPTY)
    agent.remember_cell((0, 1), CellType.EMPTY)

    clusters = find_frontier_clusters(
        agent=agent,
        width=3,
        height=3,
    )

    assert clusters
    assert clusters[0].id.startswith("frontier:")
    assert clusters[0].anchor == min(clusters[0].cells)



def test_frontier_cluster_lookup_returns_containing_cluster() -> None:
    agent = Agent(x=0, y=0)
    agent.remember_cell((1, 0), CellType.EMPTY)
    agent.remember_cell((0, 1), CellType.EMPTY)

    clusters = find_frontier_clusters(
        agent=agent,
        width=3,
        height=3,
    )
    target = min(clusters[0].cells)

    found = frontier_cluster_for_position(
        position=target,
        clusters=clusters,
    )

    assert found == clusters[0]
