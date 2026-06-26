import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def gather_with_limit(
    tasks: list[Awaitable[T]],
    limit: int,
) -> list[T]:
    """
    Execute awaitables concurrently while limiting
    the maximum number of in-flight tasks.
    """

    semaphore = asyncio.Semaphore(limit)

    async def runner(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*(runner(task) for task in tasks))
