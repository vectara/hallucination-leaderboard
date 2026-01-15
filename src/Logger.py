"""Centralized logging configuration for the HHEM Leaderboard system.

This module provides a pre-configured logger instance used throughout the
evaluation pipeline. Supports dual output to both file and console with
consistent timestamp formatting.

Functions:
    setup_logger: Create and configure a logger with file and console handlers.

Attributes:
    logger: Pre-configured Logger instance for package-wide use.

Example:
    >>> from src.Logger import logger
    >>> logger.info("Starting evaluation pipeline")
    >>> logger.warning("Rate limit encountered, retrying...")
"""

import logging
from datetime import datetime


def setup_logger(log_name: str = "log") -> logging.Logger:
    """Create and configure a logger with file and console handlers.

    Initializes a logger with INFO level that outputs to both a text file
    and the console. Uses a consistent datetime format for all log entries.
    Handlers are only added once to prevent duplicate logging.

    Args:
        log_name: Base name for the logger and output file. The log file
            will be created as "{log_name}.txt". Defaults to "log".

    Returns:
        Configured Logger instance with file and console handlers attached.

    Note:
        On first initialization, logs a CRITICAL-level startup message
        with the current timestamp to mark the beginning of a session.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(f"{log_name}.txt")
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        intro_msg = (
            f"--- {log_name} started on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )
        logger.critical(intro_msg)

    return logger


logger = setup_logger()
"""logging.Logger: Default logger instance for package-wide logging.

Automatically initialized on module import with the name "log", creating
a "log.txt" file in the current working directory. Import this instance
directly rather than calling setup_logger() for consistent logging.
"""