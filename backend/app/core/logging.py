import sys

from loguru import logger


def setup_logging():
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{module}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        "logs/omnirag.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        serialize=True,
    )


__all__ = ["logger", "setup_logging"]
