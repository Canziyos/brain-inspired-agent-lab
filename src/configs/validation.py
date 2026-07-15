def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0, got {value!r}"
        )


def require_non_empty_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")