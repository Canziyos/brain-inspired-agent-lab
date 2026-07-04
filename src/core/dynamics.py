from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics_types import ActionOutcome, EventType
from src.core.world import (
    DANGER,
    EMPTY,
    FOOD,
    MYSTERY,
    World,
)


def apply_action(
    agent: Agent,
    world: World,
    action: Action,
) -> ActionOutcome:
    old_energy = agent.energy
    old_health = agent.health
    old_curiosity = agent.curiosity

    event = EventType.NO_OP

    if action.kind == "rest":
        agent.energy += 5.0
        event = EventType.RESTED

    elif action.kind == "blocked":
        agent.energy -= 1.0
        event = EventType.BLOCKED
        
    elif action.kind == "move":
        cell = world.get_cell(
            action.target_x,
            action.target_y,
        )

        agent.move_to(
            action.target_x,
            action.target_y,
        )
        agent.energy -= 2.0

        if cell == FOOD:
            agent.energy += 25.0

            world.set_cell(
                action.target_x,
                action.target_y,
                EMPTY,
            )

            agent.known_cells[
                (action.target_x, action.target_y)
            ] = EMPTY

            event = EventType.ATE_FOOD

        elif cell == DANGER:
            agent.health -= 30.0
            agent.energy -= 5.0

            agent.known_cells[
                (action.target_x, action.target_y)
            ] = DANGER

            event = EventType.HIT_DANGER

        elif cell == MYSTERY:
            agent.curiosity -= 5.0

            world.set_cell(
                action.target_x,
                action.target_y,
                EMPTY,
            )

            agent.known_cells[
                (action.target_x, action.target_y)
            ] = EMPTY

            event = EventType.DISCOVERED_MYSTERY

        elif cell == EMPTY:
            agent.known_cells[
                (action.target_x, action.target_y)
            ] = EMPTY

            event = EventType.VISITED_EMPTY

    agent.energy = max(
        0.0,
        min(100.0, agent.energy),
    )
    agent.health = max(
        0.0,
        min(100.0, agent.health),
    )
    agent.curiosity = max(
        0.0,
        min(100.0, agent.curiosity),
    )

    outcome = ActionOutcome(
        position=(agent.x, agent.y),
        event=event,
        energy_change=agent.energy - old_energy,
        health_change=agent.health - old_health,
        curiosity_change=(
            agent.curiosity - old_curiosity
        ),
    )

    agent.record_experience(outcome)
    return outcome
