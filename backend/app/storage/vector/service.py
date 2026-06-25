from typing import Optional

from app.core.logging import logger
from app.storage.vector.base import VectorStore
from app.storage.vector.models import (
    VectorDocument,
    VectorSearchResult,
)
from app.storage.vector.qdrant import QdrantVectorStore


class VectorService:
    """
    High level vector database interface.

    Responsibilities:
    - hide vector DB implementation
    - validation
    - logging
    - future routing
    """

    def __init__(
        self,
        store: VectorStore,
    ):
        self.store = store

    async def create_collection(
        self,
        dimension: int,
    ) -> None:
        await self.store.create_collection(dimension)

    async def add_documents(
        self,
        documents: list[VectorDocument],
    ) -> None:

        if not documents:
            logger.warning("Empty vector insert skipped")
            return
        await self.store.upsert(documents)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:

        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        return await self.store.search(
            query_vector=query_vector,
            limit=limit,
        )

    async def hybrid_search(
        self,
        dense_vector,
        sparse_vector,
        limit,
    ):

        return await self.store.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

    async def delete(
        self,
        ids: list[str],
    ) -> None:

        if not ids:
            return
        await self.store.delete(ids)


_vector_service: Optional[VectorService] = None


async def init_vector_service(
    dimension: int,
):

    global _vector_service

    logger.info("Initializing vector storage")
    store = QdrantVectorStore()
    await store.connect()
    await store.create_collection(dimension)
    _vector_service = VectorService(store)
    logger.success("Vector storage ready")


async def close_vector_service():

    if _vector_service:
        await _vector_service.store.close()


def get_vector_service() -> VectorService:

    if _vector_service is None:
        raise RuntimeError("Vector service not initialized")
    return _vector_service
