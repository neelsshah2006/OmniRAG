from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """
    Public API representation of a document.
    """

    id: str
    filename: str
    status: DocumentStatus
    element_count: int
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)
