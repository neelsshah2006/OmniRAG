from abc import ABC, abstractmethod

from app.storage.vector.models import (
    VectorDocument,
    VectorSearchResult,
)


class VectorStore(ABC):
    """
    Base interface for vector databases.

    Any vector DB implementation must follow this.
    """

    @abstractmethod
    async def connect(
        self,
    ) -> None:
        pass

    @abstractmethod
    async def close(
        self,
    ) -> None:
        pass

    @abstractmethod
    async def create_collection(
        self,
        dimension: int,
    ) -> None:
        pass

    @abstractmethod
    async def upsert(
        self,
        documents: list[VectorDocument],
    ) -> None:
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        pass

    @abstractmethod
    async def delete(
        self,
        ids: list[str],
    ) -> None:
        pass
