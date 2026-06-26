import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings
from app.core.logging import logger
from app.storage.vector.base import VectorStore
from app.storage.vector.models import VectorDocument, VectorSearchResult
from app.utils.batch import batched

settings = get_settings()


class QdrantVectorStore(VectorStore):
    def __init__(self):
        self.client: AsyncQdrantClient | None = None
        self.collection = settings.QDRANT_COLLECTION

    async def connect(self) -> None:

        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        logger.info("Qdrant connected")

    async def close(self) -> None:

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
            vectors_config={
                "dense": VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        logger.success(f"Created Qdrant collection: {self.collection}")

    async def upsert(
        self,
        documents: list[VectorDocument],
    ) -> None:

        points = []
        for document in documents:
            vectors = {"dense": document.dense_vector}
            if document.sparse_vector:
                vectors["sparse"] = SparseVector(
                    indices=document.sparse_vector.indices,
                    values=document.sparse_vector.values,
                )

            points.append(
                PointStruct(
                    id=document.id,
                    vector=vectors,
                    payload=document.payload,
                )
            )

        start = time.perf_counter()

        for batch in batched(
            points,
            settings.VECTOR_UPSERT_BATCH_SIZE,
        ):
            await self.client.upsert(
                collection_name=self.collection,
                points=batch,
            )

        elapsed = time.perf_counter() - start

        logger.info(
            "Indexed {} vectors in {:.2f}s",
            len(points),
            elapsed,
        )

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using="dense",
            limit=limit,
        )

        return [
            VectorSearchResult(
                id=str(point.id),
                score=point.score,
                retrieval_type="dense",
                payload=point.payload or {},
            )
            for point in response.points
        ]

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        response = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 2,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(
                fusion=Fusion.RRF,
            ),
            limit=limit,
        )

        return [
            VectorSearchResult(
                id=str(point.id),
                score=point.score,
                retrieval_type="hybrid",
                payload=point.payload or {},
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
