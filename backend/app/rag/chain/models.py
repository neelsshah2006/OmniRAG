from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """
    Source used for generating answer.
    """

    content: str
    metadata: dict = Field(default_factory=dict)
    vector_score: float | None = None
    rerank_score: float | None = None


class RAGResponse(BaseModel):
    """
    Final RAG output returned to API/user.
    """

    answer: str
    sources: list[SourceReference]
    model: str | None = None
    usage: dict = Field(default_factory=dict)
    latency_ms: float | None = None
