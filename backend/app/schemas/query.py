from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Incoming query from the API caller.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer against ingested documents.",
    )


class SourceReferenceResponse(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    vector_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    """
    Public API representation of a RAG answer.
    """

    question: str
    answer: str
    sources: list[SourceReferenceResponse]
    model: str | None = None
    usage: dict = Field(default_factory=dict)
    latency_ms: float | None = None
