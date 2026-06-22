from typing import Any
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Chunk returned from retrieval.
    """

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """
    Final retrieval output.
    """

    query: str
    chunks: list[RetrievedChunk]
