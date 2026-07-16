from __future__ import annotations

from dataclasses import dataclass, field

from src.core.actions import Action
from src.core.dynamics_types import EventType
from src.planning.goal_planner import GoalPlan


Position = tuple[int, int]
StateVector = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Episode:
    step: int

    position_before: Position
    state_before: StateVector

    goal_kind: str | None
    goal_target: Position | None
    goal_id: str | None
    action: Action

    reward: float
    event: EventType

    position_after: Position
    state_after: StateVector

    network_action: Action
    imagination_action: Action
    choices_agree: bool
    imagination_agrees: bool


@dataclass(slots=True)
class EpisodicTrace:
    episodes: list[Episode] = field(default_factory=list)

    def record(
        self,
        episode: Episode,
    ) -> None:
        self.episodes.append(episode)

    def __len__(self) -> int:
        return len(self.episodes)



def build_episode(
    step: int,
    position_before: Position,
    state_before: StateVector,
    plan: GoalPlan | None,
    action: Action,
    reward: float,
    event: EventType,
    position_after: Position,
    state_after: StateVector,
    network_action: Action,
    imagination_action: Action,
    choices_agree: bool,
    imagination_agrees: bool,
) -> Episode:
    return Episode(
        step=step,
        position_before=position_before,
        state_before=state_before,
        goal_kind=(
            plan.kind.value
            if plan is not None
            else None
        ),
        goal_target=(
            plan.target
            if plan is not None
            else None
        ),
        goal_id=(
            plan.goal_id
            if plan is not None
            else None
        ),
        action=action,
        reward=reward,
        event=event,
        position_after=position_after,
        state_after=state_after,
        network_action=network_action,
        imagination_action=imagination_action,
        choices_agree=choices_agree,
        imagination_agrees=imagination_agrees,
    )
