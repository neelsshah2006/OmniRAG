from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentElement(BaseModel):
    """
    Smallest extracted unit from a source document.

    Created by loaders/parsers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    type: str
    # Title
    # NarrativeText
    # Table
    # Image
    content: str | None = None
    element_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_element: Any | None = None


class Document(BaseModel):
    """
    Parsed source document.
    """

    id: str
    elements: list[DocumentElement]
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
    element_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
