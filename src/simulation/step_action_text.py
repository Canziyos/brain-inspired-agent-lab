from src.core.actions import Action


def format_action(action: Action) -> str:
    if action.is_rest:
        return "rest"

    dx, dy = action.delta

    return (
        f"move {action.name.lower()} "
        f"({dx:+d}, {dy:+d})"
    )