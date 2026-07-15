from typing import Final

from src.core.dynamics_types import EventType


EVENT_TYPES: Final[tuple[EventType, ...]] = tuple(EventType)

EVENT_TO_INDEX: Final[dict[EventType, int]] = {
    event: index
    for index, event in enumerate(EVENT_TYPES)
}


def event_index(event: EventType) -> int:
    return EVENT_TO_INDEX[event]