from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.core.world import CellType
from src.memory.working_memory import WorkingMemory
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
    GoalPreference,
    choose_preferred_goal,
)
from src.planning.goal_scoring import (
    GoalScore,
    ScoredGoalPlan,
)



def scored_plan(
    kind: GoalKind,
    target: tuple[int, int],
    total: float,
    goal_id: str | None = None,
) -> ScoredGoalPlan:
    plan = GoalPlan(
        kind=kind,
        target=target,
        path=((0, 0), target),
        goal_id=goal_id,
    )

    return ScoredGoalPlan(
        plan=plan,
        score=GoalScore(
            motivation=0.0,
            information_gain=0.0,
            energy_bonus=0.0,
            distance_cost=0.0,
            danger_risk=0.0,
            energy_risk=0.0,
            total=total,
        ),
    )



def test_working_memory_tracks_selected_goal_and_switches() -> None:
    memory = WorkingMemory()

    first = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        path=((0, 0), (1, 0)),
    )
    second = GoalPlan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        path=((1, 0), (2, 0)),
    )

    memory.remember_selected_goal(first)
    memory.remember_selected_goal(first)
    memory.remember_selected_goal(second)

    snapshot = memory.snapshot()

    assert snapshot.current_goal_kind == "mystery"
    assert snapshot.current_goal_target == (2, 0)
    assert snapshot.current_goal_age == 1
    assert snapshot.goal_switch_count == 1
    assert snapshot.target_switch_count == 1



def test_working_memory_goal_preference_uses_current_goal() -> None:
    memory = WorkingMemory()
    plan = GoalPlan(
        kind=GoalKind.FOOD,
        target=(1, 0),
        path=((0, 0), (1, 0)),
    )

    memory.remember_selected_goal(plan)

    preference = memory.goal_preference()

    assert preference is not None
    assert preference.kind is GoalKind.FOOD
    assert preference.target == (1, 0)
    assert preference.goal_id == "food:1:0"
    assert preference.continuation_bonus > 0.0
    assert preference.switch_margin > 0.0



def test_frontier_goal_identity_tracks_region_not_exact_target() -> None:
    memory = WorkingMemory()

    first = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        path=((0, 0), (1, 0)),
        goal_id="frontier:0:0",
    )
    second = GoalPlan(
        kind=GoalKind.FRONTIER,
        target=(1, 1),
        path=((1, 0), (1, 1)),
        goal_id="frontier:0:0",
    )

    memory.remember_selected_goal(first)
    memory.remember_selected_goal(second)

    snapshot = memory.snapshot()

    assert snapshot.current_goal_kind == "frontier"
    assert snapshot.current_goal_target == (1, 1)
    assert snapshot.current_goal_id == "frontier:0:0"
    assert snapshot.current_goal_age == 2
    assert snapshot.goal_switch_count == 0
    assert snapshot.target_switch_count == 1
    assert snapshot.frontier_target_switch_count == 1
    assert snapshot.frontier_semantic_switch_count == 0



def test_goal_preference_keeps_current_goal_without_clear_win() -> None:
    current = scored_plan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        total=10.0,
    )
    challenger = scored_plan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        total=20.0,
    )

    choice = choose_preferred_goal(
        scored_plans=(current, challenger),
        preference=GoalPreference(
            kind=GoalKind.FRONTIER,
            target=(1, 0),
            continuation_bonus=8.0,
            switch_margin=6.0,
        ),
    )

    assert choice.plan.kind is GoalKind.FRONTIER
    assert choice.total == 18.0



def test_goal_preference_matches_frontier_by_goal_id() -> None:
    current = scored_plan(
        kind=GoalKind.FRONTIER,
        target=(1, 1),
        total=10.0,
        goal_id="frontier:0:0",
    )
    challenger = scored_plan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        total=20.0,
    )

    choice = choose_preferred_goal(
        scored_plans=(current, challenger),
        preference=GoalPreference(
            kind=GoalKind.FRONTIER,
            target=(1, 0),
            continuation_bonus=8.0,
            switch_margin=6.0,
            goal_id="frontier:0:0",
        ),
    )

    assert choice.plan.kind is GoalKind.FRONTIER
    assert choice.plan.target == (1, 1)
    assert choice.total == 18.0



def test_goal_preference_allows_clear_win() -> None:
    current = scored_plan(
        kind=GoalKind.FRONTIER,
        target=(1, 0),
        total=10.0,
    )
    challenger = scored_plan(
        kind=GoalKind.MYSTERY,
        target=(2, 0),
        total=30.0,
    )

    choice = choose_preferred_goal(
        scored_plans=(current, challenger),
        preference=GoalPreference(
            kind=GoalKind.FRONTIER,
            target=(1, 0),
            continuation_bonus=8.0,
            switch_margin=6.0,
        ),
    )

    assert choice.plan.kind is GoalKind.MYSTERY
    assert choice.total == 30.0



def test_working_memory_records_recent_step_context() -> None:
    memory = WorkingMemory()
    agent = Agent(x=2, y=3, energy=44.0)

    memory.record_step(
        agent=agent,
        action=Action.REST,
        reward=-1.0,
        event=EventType.RESTED,
    )

    snapshot = memory.snapshot()

    assert snapshot.recent_rest_count == 1
    assert snapshot.recent_position_revisit_count == 0
    assert snapshot.energy_trend == 0.0



def test_working_memory_tracks_coverage_and_reachable_frontiers() -> None:
    memory = WorkingMemory()
    agent = Agent(x=0, y=0)
    agent.remember_cell((1, 0), CellType.EMPTY)
    agent.remember_cell((0, 1), CellType.EMPTY)

    memory.update_coverage(
        step=0,
        agent=agent,
        width=3,
        height=3,
    )

    snapshot = memory.snapshot()

    assert snapshot.total_world_cells == 9
    assert snapshot.seen_cell_count == 3
    assert snapshot.visited_cell_count == 1
    assert snapshot.unseen_cell_count == 6
    assert snapshot.frontier_count > 0
    assert snapshot.reachable_frontier_count > 0
    assert snapshot.frontier_cluster_count >= 1
    assert snapshot.reachable_frontier_cluster_count >= 1
