from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(
    items: list[T],
    batch_size: int,
) -> Iterator[list[T]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
