import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.logging import logger

T = TypeVar("T")


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    retries: int = 3,
    base_delay: float = 1.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    operation: str = "Operation",
    **kwargs,
) -> T:
    """
    Execute an async function with exponential backoff.

    Example delays (base_delay=1):
        Attempt 1 -> 1s
        Attempt 2 -> 2s
        Attempt 3 -> 4s

    A small random jitter is added to reduce retry storms.
    """

    for attempt in range(retries + 1):
        try:
            return await func(*args, **kwargs)

        except retry_exceptions as error:
            if attempt == retries:
                logger.exception(
                    "{} failed after {} retries",
                    operation,
                    retries,
                )
                raise

            delay = min(
                base_delay * (2**attempt),
                30,
            )
            delay += random.uniform(0, 0.5)

            logger.warning(
                "{} failed (attempt {}/{}). Retrying in {:.2f}s",
                operation,
                attempt + 1,
                retries,
                delay,
            )

            await asyncio.sleep(delay)
