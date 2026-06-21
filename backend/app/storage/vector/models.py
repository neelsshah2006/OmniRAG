from pydantic import BaseModel, Field
from typing import Any


class VectorDocument(BaseModel):
    """
    Document chunk stored in vector DB.
    """

    id: str
    vector: list[float]
    payload: dict[str, Any]


class VectorSearchResult(BaseModel):

    id: str
    score: float
    payload: dict[str, Any]
