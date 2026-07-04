from src.core.actions import Action
from src.core.agent import Agent
from src.core.dynamics import apply_action
from src.core.dynamics_types import EventType
from src.core.world import EMPTY, FOOD, World


def make_action(name: str, x: int, y: int) -> Action:
    return Action(
        kind=name,
        target_x=x,
        target_y=y,
    )


def test_moving_to_food_updates_agent_world_and_experience() -> None:
    world = World(
        width=2,
        height=1,
        grid=[[EMPTY, FOOD]],
    )
    agent = Agent(x=0, y=0, energy=70.0)

    outcome = apply_action(
        agent,
        world,
        make_action("move", 1, 0),
    )

    assert (agent.x, agent.y) == (1, 0)
    assert agent.energy == 93.0
    assert world.get_cell(1, 0) == EMPTY
    assert agent.known_cells[(1, 0)] == EMPTY
    assert agent.experiences[-1].position == (1, 0)
    assert agent.experiences[-1].energy_change == 23.0
    assert outcome.event == EventType.ATE_FOOD


def test_rest_clamps_energy_at_maximum() -> None:
    world = World(width=1, height=1)
    agent = Agent(x=0, y=0, energy=98.0)

    outcome = apply_action(
        agent,
        world,
        make_action("rest", 0, 0),
    )

    assert agent.energy == 100.0
    assert agent.experiences[-1].energy_change == 2.0
    assert outcome.event == EventType.RESTED
