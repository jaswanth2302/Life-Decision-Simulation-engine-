"""
Generative Agent Simulation Engine — Structured Logging Configuration
=====================================================================

Configures a robust, structured logging setup using Python's standard library
for tracking simulation pipeline ticks. Outputs clean, timestamped console logs
with configurable verbosity via the SIMULATION_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_LEVEL: str = "INFO"
_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
_ROOT_LOGGER_NAME: str = "simulation_engine"


def configure_logging(
    level: str | None = None,
    *,
    stream: object = None,
) -> logging.Logger:
    """Configure and return the root simulation engine logger.

    Parameters
    ----------
    level : str | None
        Logging level string (e.g. ``"DEBUG"``, ``"INFO"``). Falls back to
        the ``SIMULATION_LOG_LEVEL`` environment variable, then to ``"INFO"``.
    stream : object, optional
        Output stream for the console handler. Defaults to ``sys.stdout``.

    Returns
    -------
    logging.Logger
        The configured root logger for the simulation engine.
    """
    resolved_level: str = (
        level
        or os.environ.get("SIMULATION_LOG_LEVEL")
        or _DEFAULT_LOG_LEVEL
    ).upper()

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(resolved_level)

    # Avoid duplicate handlers on repeated calls.
    if not logger.handlers:
        console_handler = logging.StreamHandler(stream or sys.stdout)
        console_handler.setLevel(resolved_level)

        formatter = logging.Formatter(
            fmt=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger scoped to the given module.

    Parameters
    ----------
    module_name : str
        Fully qualified module name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        A child logger under the simulation engine namespace.
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{module_name}")
