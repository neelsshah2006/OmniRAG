from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Parsed source document.
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """
    Smaller searchable document unit.
    """

    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
