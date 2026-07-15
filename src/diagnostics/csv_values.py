from enum import Enum


def optional_csv(value: object | None) -> object:
    if value is None:
        return ""

    return value


def enum_csv(value: object) -> object:
    if isinstance(value, Enum):
        return value.value

    return value