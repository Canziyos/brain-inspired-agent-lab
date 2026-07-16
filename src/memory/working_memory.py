from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Deque, TypeAlias

from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.planning.frontier_planner import find_frontiers
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
    GoalPreference,
)


Position: TypeAlias = tuple[int, int]

RECENT_POSITION_LIMIT = 12
RECENT_ACTION_LIMIT = 12
RECENT_REWARD_LIMIT = 8
RECENT_ENERGY_LIMIT = 8
RECENT_EVENT_LIMIT = 12

GOAL_CONTINUATION_BONUS = 8.0
GOAL_SWITCH_MARGIN = 6.0

STUCK_WINDOW = 6
STUCK_UNIQUE_POSITION_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class WorkingMemorySnapshot:
    current_goal_kind: str | None
    current_goal_target: Position | None
    current_goal_age: int
    goal_switch_count: int

    recent_position_revisit_count: int
    stuck_counter: int
    recent_rest_count: int
    energy_trend: float

    last_food_position: Position | None
    last_mystery_position: Position | None
    last_danger_position: Position | None

    total_world_cells: int
    seen_cell_count: int
    visited_cell_count: int
    unseen_cell_count: int
    seen_ratio: float
    visited_ratio: float
    frontier_count: int
    newly_seen_count: int
    newly_visited_count: int
    first_full_seen_step: int | None
    first_full_visited_step: int | None


@dataclass(slots=True)
class WorkingMemory:
    recent_positions: Deque[Position] = field(
        default_factory=lambda: deque(maxlen=RECENT_POSITION_LIMIT)
    )
    recent_actions: Deque[Action] = field(
        default_factory=lambda: deque(maxlen=RECENT_ACTION_LIMIT)
    )
    recent_rewards: Deque[float] = field(
        default_factory=lambda: deque(maxlen=RECENT_REWARD_LIMIT)
    )
    recent_energy: Deque[float] = field(
        default_factory=lambda: deque(maxlen=RECENT_ENERGY_LIMIT)
    )
    recent_events: Deque[EventType] = field(
        default_factory=lambda: deque(maxlen=RECENT_EVENT_LIMIT)
    )

    current_goal_kind: GoalKind | None = None
    current_goal_target: Position | None = None
    current_goal_age: int = 0
    goal_switch_count: int = 0

    stuck_counter: int = 0

    last_food_position: Position | None = None
    last_mystery_position: Position | None = None
    last_danger_position: Position | None = None

    first_seen_step: dict[Position, int] = field(default_factory=dict)
    first_visited_step: dict[Position, int] = field(default_factory=dict)
    total_world_cells: int = 0
    frontier_count: int = 0
    newly_seen_count: int = 0
    newly_visited_count: int = 0
    first_full_seen_step: int | None = None
    first_full_visited_step: int | None = None

    def goal_preference(self) -> GoalPreference | None:
        if (
            self.current_goal_kind is None
            or self.current_goal_target is None
        ):
            return None

        return GoalPreference(
            kind=self.current_goal_kind,
            target=self.current_goal_target,
            continuation_bonus=GOAL_CONTINUATION_BONUS,
            switch_margin=GOAL_SWITCH_MARGIN,
        )

    def remember_selected_goal(
        self,
        plan: GoalPlan | None,
    ) -> None:
        if plan is None:
            self.current_goal_kind = None
            self.current_goal_target = None
            self.current_goal_age = 0
            return

        if self._matches_current_goal(plan):
            self.current_goal_age += 1
            return

        if self.current_goal_kind is not None:
            self.goal_switch_count += 1

        self.current_goal_kind = plan.kind
        self.current_goal_target = plan.target
        self.current_goal_age = 1

    def record_step(
        self,
        agent: Agent,
        action: Action,
        reward: float,
        event: EventType,
    ) -> None:
        self.recent_positions.append(agent.position)
        self.recent_actions.append(action)
        self.recent_rewards.append(float(reward))
        self.recent_energy.append(float(agent.energy))
        self.recent_events.append(event)

        self._remember_event_position(
            event=event,
            position=agent.position,
        )
        self._update_stuck_counter()

    def update_coverage(
        self,
        step: int,
        agent: Agent,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("World dimensions must be positive.")

        self.total_world_cells = width * height

        seen_positions = set(agent.known_cells) | set(agent.visited)
        visited_positions = set(agent.visited)

        self.newly_seen_count = remember_first_steps(
            first_steps=self.first_seen_step,
            positions=seen_positions,
            step=step,
        )
        self.newly_visited_count = remember_first_steps(
            first_steps=self.first_visited_step,
            positions=visited_positions,
            step=step,
        )

        self.frontier_count = len(
            find_frontiers(
                agent=agent,
                width=width,
                height=height,
            )
        )

        if (
            self.first_full_seen_step is None
            and len(self.first_seen_step) >= self.total_world_cells
        ):
            self.first_full_seen_step = step

        if (
            self.first_full_visited_step is None
            and len(self.first_visited_step) >= self.total_world_cells
        ):
            self.first_full_visited_step = step

    def snapshot(self) -> WorkingMemorySnapshot:
        seen_cell_count = len(self.first_seen_step)
        visited_cell_count = len(self.first_visited_step)
        unseen_cell_count = max(
            0,
            self.total_world_cells - seen_cell_count,
        )

        return WorkingMemorySnapshot(
            current_goal_kind=(
                self.current_goal_kind.value
                if self.current_goal_kind is not None
                else None
            ),
            current_goal_target=self.current_goal_target,
            current_goal_age=self.current_goal_age,
            goal_switch_count=self.goal_switch_count,
            recent_position_revisit_count=(
                recent_duplicate_count(self.recent_positions)
            ),
            stuck_counter=self.stuck_counter,
            recent_rest_count=sum(
                1
                for event in self.recent_events
                if event is EventType.RESTED
            ),
            energy_trend=sequence_trend(self.recent_energy),
            last_food_position=self.last_food_position,
            last_mystery_position=self.last_mystery_position,
            last_danger_position=self.last_danger_position,
            total_world_cells=self.total_world_cells,
            seen_cell_count=seen_cell_count,
            visited_cell_count=visited_cell_count,
            unseen_cell_count=unseen_cell_count,
            seen_ratio=safe_ratio(
                seen_cell_count,
                self.total_world_cells,
            ),
            visited_ratio=safe_ratio(
                visited_cell_count,
                self.total_world_cells,
            ),
            frontier_count=self.frontier_count,
            newly_seen_count=self.newly_seen_count,
            newly_visited_count=self.newly_visited_count,
            first_full_seen_step=self.first_full_seen_step,
            first_full_visited_step=self.first_full_visited_step,
        )

    def _matches_current_goal(
        self,
        plan: GoalPlan,
    ) -> bool:
        return (
            self.current_goal_kind is plan.kind
            and self.current_goal_target == plan.target
        )

    def _remember_event_position(
        self,
        event: EventType,
        position: Position,
    ) -> None:
        if event is EventType.ATE_FOOD:
            self.last_food_position = position

        elif event is EventType.DISCOVERED_MYSTERY:
            self.last_mystery_position = position

        elif event is EventType.HIT_DANGER:
            self.last_danger_position = position

    def _update_stuck_counter(self) -> None:
        recent = tuple(self.recent_positions)[-STUCK_WINDOW:]

        if len(recent) < STUCK_WINDOW:
            self.stuck_counter = 0
            return

        if len(set(recent)) <= STUCK_UNIQUE_POSITION_THRESHOLD:
            self.stuck_counter += 1
            return

        self.stuck_counter = 0


def recent_duplicate_count(
    positions: Iterable[Position],
) -> int:
    seen: set[Position] = set()
    duplicates = 0

    for position in positions:
        if position in seen:
            duplicates += 1
        else:
            seen.add(position)

    return duplicates


def sequence_trend(
    values: Iterable[float],
) -> float:
    sequence = tuple(values)

    if len(sequence) < 2:
        return 0.0

    return sequence[-1] - sequence[0]


def remember_first_steps(
    first_steps: dict[Position, int],
    positions: Iterable[Position],
    step: int,
) -> int:
    new_count = 0

    for position in positions:
        if position in first_steps:
            continue

        first_steps[position] = step
        new_count += 1

    return new_count


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator
