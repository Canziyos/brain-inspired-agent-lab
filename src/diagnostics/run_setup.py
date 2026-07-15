import logging
from datetime import datetime
from pathlib import Path
from typing import Final


LOGGER_NAME: Final[str] = "brain_lab"


def create_run_directory(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_directory = Path(output_root) / timestamp

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def configure_run_logging(
    run_directory: Path | None,
    debug: bool,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.DEBUG
        if debug
        else logging.INFO
    )
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )

    logger.addHandler(console_handler)

    if run_directory is None:
        return logger

    file_handler = logging.FileHandler(
        run_directory / "run.log",
        encoding="utf-8",
    )
    file_handler.setLevel(
        logging.DEBUG
        if debug
        else logging.INFO
    )
    file_handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(file_handler)

    return logger