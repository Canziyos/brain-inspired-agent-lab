from src.core.agent import Agent
from src.core.world import CellType
from src.diagnostics.coverage_csv import coverage_rows
from src.memory.working_memory import WorkingMemory


def test_coverage_rows_include_first_seen_and_visited_steps() -> None:
    memory = WorkingMemory()
    agent = Agent(x=0, y=0)
    agent.known_cells[(1, 0)] = CellType.EMPTY

    memory.update_coverage(
        step=4,
        agent=agent,
        width=2,
        height=1,
    )

    rows = coverage_rows(
        memory=memory,
        width=2,
        height=1,
    )

    assert rows == [
        {
            "x": 0,
            "y": 0,
            "seen": True,
            "visited": True,
            "first_seen_step": 4,
            "first_visited_step": 4,
        },
        {
            "x": 1,
            "y": 0,
            "seen": True,
            "visited": False,
            "first_seen_step": 4,
            "first_visited_step": "",
        },
    ]
