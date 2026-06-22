from app.core.logging import logger
from app.ai.embeddings.service import get_embedding_service
from app.storage.vector.service import get_vector_service
from app.rag.retrieval.models import (
    RetrievedChunk,
    RetrievalResult,
)


class Retriever:
    """
    Retrieves relevant knowledge for a query.
    """

    async def retrieve(self, query: str, limit: int = 5) -> RetrievalResult:
        logger.info("Starting retrieval")
        embedding_service = get_embedding_service()
        vector_service = get_vector_service()

        # 1. Convert Question into Vector
        query_vector = await embedding_service.embed_text(query)

        # 2. Semantic Search
        results = await vector_service.search(query_vector=query_vector, limit=limit)

        chunks: list[RetrievedChunk] = []
        # 3. Convert Storage results into retrieval objects
        for result in results:
            chunks.append(
                RetrievedChunk(
                    id=result.id,
                    content=result.payload.get("text", ""),
                    score=result.score,
                    metadata={
                        key: value
                        for key, value in result.payload.items()
                        if key != "text"
                    },
                )
            )

        logger.info(f"Retrieved {len(chunks)} chunks")

        return RetrievalResult(query=query, chunks=chunks)


retriever = Retriever()
