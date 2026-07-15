import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "adfusion",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Sets up a logger with handlers for console and an optional file.

    Args:
        name: The name of the logger.
        log_file: The path to the file where logs should be written.
        level: The logging level (e.g. logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if the logger is re-initialized
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (if file path is provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
