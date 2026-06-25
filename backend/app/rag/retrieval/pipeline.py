import time

from app.core.config import get_settings
from app.core.logging import logger

from app.rag.retrieval.models import RetrievalResult
from app.rag.retrieval.service import retrieval_service
from app.rag.query.service import query_service

from app.ai.reranker.service import reranker_service

settings = get_settings()


class RetrievalPipeline:
    """
    Production retrieval workflow.

    Query
      ↓
    Retrieve candidates
      ↓
    Rerank
      ↓
    Return best context
    """

    async def retrieve(self, query: str) -> RetrievalResult:
        start = time.perf_counter()

        # 1. Process Query
        processed_query = await query_service.process(query)

        # 2. Retrieve candidates
        result = await retrieval_service.retrieve(query=processed_query)
        if not result.chunks:
            return result

        # 3. Prepare reranker input
        documents = [chunk.content for chunk in result.chunks]

        reranked = await reranker_service.rerank(
            query=processed_query,
            documents=documents,
        )

        # 4. Select top reranked chunks
        selected_chunks = []
        for item in reranked[: settings.RERANK_TOP_K]:
            chunk = result.chunks[item.index]
            chunk.rerank_score = item.score
            selected_chunks.append(chunk)

        latency = (time.perf_counter() - start) * 1000
        logger.info(
            f"Retrieval pipeline completed: "
            f"{len(result.chunks)} → "
            f"{len(selected_chunks)} chunks "
            f"({latency:.2f}ms)"
        )

        return RetrievalResult(query=query, chunks=selected_chunks)


retrieval_pipeline = RetrievalPipeline()
