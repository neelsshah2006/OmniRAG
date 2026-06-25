import asyncio

from app.ai.embeddings.service import get_embedding_service
from app.ai.sparse.service import sparse_embedding_service
from app.core.config import get_settings
from app.core.logging import logger
from app.rag.retrieval.models import RetrievalResult, RetrievedChunk
from app.rag.retrieval.retrievers.base import BaseRetriever
from app.storage.vector.service import get_vector_service

settings = get_settings()


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval:
    Dense semantic search
    +
    Sparse keyword search
    """

    async def retrieve(self, query: str) -> RetrievalResult:

        logger.info("Running hybrid retrieval")

        embedding_service = get_embedding_service()
        vector_service = get_vector_service()

        dense, sparse = await asyncio.gather(
            embedding_service.embed_text(query),
            sparse_embedding_service.embed(query),
        )

        results = await vector_service.hybrid_search(
            dense_vector=dense,
            sparse_vector=sparse,
            limit=settings.RETRIEVAL_TOP_K,
        )

        chunks: list[RetrievedChunk] = []

        for result in results:
            chunks.append(
                RetrievedChunk(
                    id=result.id,
                    content=result.payload.get(
                        "text",
                        "",
                    ),
                    vector_score=result.score,
                    metadata={
                        key: value
                        for key, value in result.payload.items()
                        if key != "text"
                    },
                )
            )

        logger.info(f"Hybrid retrieval returned {len(chunks)} chunks")

        return RetrievalResult(
            query=query,
            chunks=chunks,
        )
