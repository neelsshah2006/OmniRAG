from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_event import (
    DocumentEvent,
    DocumentEventType,
)


class DocumentEventRepository:
    """
    Data access layer for document lifecycle events.

    Responsibilities:
    - Store document processing events
    - Retrieve document history timeline

    Does NOT:
    - Change document status
    - Execute ingestion logic
    """

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

    async def create(
        self,
        document_id: str,
        event_type: DocumentEventType,
        message: str | None = None,
        metadata: dict | None = None,
    ) -> DocumentEvent:
        """
        Create a new immutable lifecycle event.

        Example:
        Document uploaded
        Parsing started
        Chunking completed
        Indexing completed
        """

        event = DocumentEvent(
            document_id=document_id,
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        )

        self.session.add(event)
        await self.session.flush()
        return event

    async def get_by_document(
        self,
        document_id: str,
    ) -> DocumentEvent:
        """
        Retrieve complete processing timeline
        for a document.
        """

        result = await self.session.execute(
            select(DocumentEvent)
            .where(DocumentEvent.document_id == document_id)
            .order_by(DocumentEvent.created_at.asc())
        )

        return list(result.scalars().all())
