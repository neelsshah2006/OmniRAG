from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.core.config import get_settings
from app.core.logging import logger
from app.storage.vector.base import VectorStore
from app.storage.vector.models import (
    VectorDocument,
    VectorSearchResult,
)

settings = get_settings()


class QdrantVectorStore(VectorStore):
    def __init__(self):
        self.client: AsyncQdrantClient | None = None
        self.collection = settings.QDRANT_COLLECTION

    async def connect(
        self,
    ) -> None:

        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        logger.info("Qdrant connected")

    async def close(
        self,
    ) -> None:

        if self.client:
            await self.client.close()
            logger.info("Qdrant disconnected")

    async def create_collection(
        self,
        dimension: int,
    ) -> None:

        exists = await self.client.collection_exists(self.collection)
        if exists:
            logger.info(f"Qdrant collection exists: {self.collection}")
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

        logger.success(f"Created Qdrant collection: {self.collection}")

    async def upsert(
        self,
        documents: list[VectorDocument],
    ) -> None:

        points = [
            PointStruct(
                id=document.id,
                vector=document.vector,
                payload=document.payload,
            )
            for document in documents
        ]

        await self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

        logger.info(f"Inserted {len(points)} vectors")

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
        )

        return [
            VectorSearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in response.points
        ]

    async def delete(
        self,
        ids: list[str],
    ) -> None:

        await self.client.delete(
            collection_name=self.collection,
            points_selector=ids,
        )
