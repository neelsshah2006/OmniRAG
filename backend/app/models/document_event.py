import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.storage.database.base import Base
from app.utils.time import utc_now

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentEventType(str, Enum):
    """
    Events generated during document lifecycle.

    Unlike Document.status, events represent
    historical processing steps.
    """

    UPLOADED = "uploaded"

    PROCESSING_STARTED = "processing_started"

    PARSING_STARTED = "parsing_started"
    PARSING_COMPLETED = "parsing_completed"

    ENRICHMENT_STARTED = "enrichment_started"
    ENRICHMENT_COMPLETED = "enrichment_completed"

    CHUNKING_COMPLETED = "chunking_completed"

    EMBEDDING_COMPLETED = "embedding_completed"

    INDEXING_COMPLETED = "indexing_completed"

    COMPLETED = "completed"

    FAILED = "failed"


class DocumentEvent(Base):
    """
    Immutable audit record for document processing.

    Used for:
    - debugging ingestion failures
    - showing progress timeline
    - monitoring pipeline performance
    """

    __tablename__ = "document_events"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[DocumentEventType] = mapped_column(
        SQLEnum(DocumentEventType),
        nullable=False,
        index=True,
    )

    message: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    event_metadata: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )

    document: Mapped["Document"] = relationship(
        back_populates="events",
    )
