from pydantic import BaseModel, Field
from typing import Any


class StoredSparseVector(BaseModel):
    """
    Sparse vector representation.

    Used for:
    - BM25
    - SPLADE
    - keyword retrieval
    """

    indices: list[int]
    values: list[float]


class VectorDocument(BaseModel):
    """
    Document stored inside vector database.

    Supports:
    - dense embeddings
    - sparse embeddings
    """

    id: str
    dense_vector: list[float]
    sparse_vector: StoredSparseVector | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """
    Search result returned from vector storage.

    Score meaning depends on retrieval mode:
    - dense similarity score
    - sparse BM25 score
    - hybrid fusion score
    """

    id: str
    score: float
    retrieval_type: str = "dense"
    payload: dict[str, Any] = Field(default_factory=dict)
