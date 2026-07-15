from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics_types import ActionOutcome, EventType
from src.core.world import CellType, World


REST_ENERGY_GAIN = 5.0
BLOCKED_ENERGY_COST = 1.0

MOVE_ENERGY_COST = 2.0
FOOD_ENERGY_GAIN = 25.0

DANGER_HEALTH_COST = 30.0
DANGER_ENERGY_COST = 5.0

MYSTERY_CURIOSITY_COST = 5.0


def _resolve_action(
    agent: Agent,
    world: World,
    action: Action,
) -> EventType:
    if action.is_rest:
        agent.energy += REST_ENERGY_GAIN
        return EventType.RESTED

    target_x, target_y = action.target_from(agent.position)

    if not world.is_inside(target_x, target_y):
        agent.energy -= BLOCKED_ENERGY_COST
        return EventType.BLOCKED

    cell = world.get_cell(target_x, target_y)
    if cell is None:
        raise ValueError(
            "World returned no cell for an inside position."
        )

    agent.move_to(target_x, target_y)
    agent.energy -= MOVE_ENERGY_COST

    if cell is CellType.FOOD:
        agent.energy += FOOD_ENERGY_GAIN
        world.set_cell(target_x, target_y, CellType.EMPTY)
        agent.remember_cell(
            position=(target_x, target_y),
            cell=CellType.EMPTY,
        )
        return EventType.ATE_FOOD

    if cell is CellType.DANGER:
        agent.health -= DANGER_HEALTH_COST
        agent.energy -= DANGER_ENERGY_COST
        agent.remember_cell(
            position=(target_x, target_y),
            cell=CellType.DANGER,
        )
        return EventType.HIT_DANGER

    if cell is CellType.MYSTERY:
        agent.curiosity -= MYSTERY_CURIOSITY_COST
        world.set_cell(target_x, target_y, CellType.EMPTY)
        agent.remember_cell(
            position=(target_x, target_y),
            cell=CellType.EMPTY,
        )
        return EventType.DISCOVERED_MYSTERY

    if cell is CellType.EMPTY:
        agent.remember_cell(
            position=(target_x, target_y),
            cell=CellType.EMPTY,
        )
        return EventType.VISITED_EMPTY

    raise ValueError(f"Unhandled cell type: {cell!r}")


def apply_action(
    agent: Agent,
    world: World,
    action: Action,
) -> ActionOutcome:
    old_energy = agent.energy
    old_health = agent.health
    old_curiosity = agent.curiosity

    event = _resolve_action(
        agent=agent,
        world=world,
        action=action,
    )

    agent.clamp_state()

    outcome = ActionOutcome(
        new_position=agent.position,
        event=event,
        energy_change=agent.energy - old_energy,
        health_change=agent.health - old_health,
        curiosity_change=agent.curiosity - old_curiosity,
    )

    agent.record_experience(outcome)
    return outcome