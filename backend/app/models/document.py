import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database.base import Base
from app.utils.time import utc_now

if TYPE_CHECKING:
    from app.models.document_event import DocumentEvent


class DocumentStatus(str, Enum):
    """
    Current lifecycle state of a document.

    Represents the latest state only.
    Detailed processing history is tracked separately
    through document events/logs.
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class Document(Base):
    """
    Persistent representation of an uploaded document.

    Responsibilities:
    - Track uploaded source files
    - Maintain ingestion lifecycle status
    - Store processing statistics
    - Link relational data with vector indexes

    Note:
    Actual document chunks and embeddings are stored
    separately in the vector database.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    filename: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    file_path: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    file_size: Mapped[int | None] = mapped_column(nullable=True)

    content_type: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus),
        default=DocumentStatus.UPLOADED,
        index=True,
    )

    element_count: Mapped[int] = mapped_column(default=0)

    chunk_count: Mapped[int] = mapped_column(default=0)

    error_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    document_metadata: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    events: Mapped[list["DocumentEvent"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
