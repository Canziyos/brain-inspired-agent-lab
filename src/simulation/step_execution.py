from dataclasses import dataclass
from typing import Any

from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics_types import EventType
from src.envs.grid_world_env import (
    BabyViceGridEnv,
    action_to_discrete,
)
from src.learning.types import StateDelta


@dataclass(frozen=True, slots=True)
class StepExecution:
    reward: float
    terminated: bool
    truncated: bool

    event: EventType
    state_delta: StateDelta

    grid_snapshot: Any
    termination_reason: str | None


def execute_step_action(
    env: BabyViceGridEnv,
    agent: Agent,
    action: Action,
) -> StepExecution:
    discrete_action = action_to_discrete(
        position=agent.position,
        action=action,
    )

    (
        _observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(discrete_action)

    event = EventType[
        info["event"]
    ]

    state_delta: StateDelta = (
        float(info["energy_change"]),
        float(info["health_change"]),
        float(info["curiosity_change"]),
    )

    return StepExecution(
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        event=event,
        state_delta=state_delta,
        grid_snapshot=info["grid"],
        termination_reason=info.get(
            "termination_reason"
        ),
    )