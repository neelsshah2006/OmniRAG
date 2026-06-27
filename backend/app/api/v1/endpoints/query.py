from fastapi import APIRouter

from app.rag.chain.service import rag_chain
from app.schemas.query import QueryRequest, QueryResponse, SourceReferenceResponse

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question against the ingested documents.

    Flow:
    Rewrite query → Retrieve → Rerank → Build context → Generate answer
    """
    result = await rag_chain.ask(question=request.question)

    return QueryResponse(
        question=request.question,
        answer=result.answer,
        sources=[
            SourceReferenceResponse(
                content=s.content,
                metadata=s.metadata,
                vector_score=s.vector_score,
                rerank_score=s.rerank_score,
            )
            for s in result.sources
        ],
        model=result.model,
        usage=result.usage,
        latency_ms=result.latency_ms,
    )
