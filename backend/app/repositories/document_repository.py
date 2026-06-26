from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    Document,
    DocumentStatus,
)


class DocumentRepository:
    """
    Data access layer for Document entities.

    Responsibilities:
    - Execute database queries
    - Persist document state changes
    - Hide SQLAlchemy details from services

    Does NOT contain:
    - Business logic
    - File processing
    - RAG ingestion logic
    """

    def __init__(
        self,
        session: AsyncSession,
    ):
        """
        Repository works within the lifecycle
        of a single database session.
        """

        self.session = session

    async def create(
        self,
        document: Document,
    ) -> Document:
        """
        Persist a newly created document record.

        Used when a document is uploaded before
        starting the ingestion pipeline.
        """

        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(
        self,
        document_id: str,
    ) -> Document | None:
        """
        Fetch a single document using its ID.

        Returns None when the document does not exist.
        """

        results = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )

        return results.scalar_one_or_none()

    async def get_by_content_hash(
        self,
        content_hash: str,
    ) -> Document | None:
        """
        Retrieve a document by its content hash.

        Used to detect duplicate uploads before
        starting the ingestion pipeline.
        """

        result = await self.session.execute(
            select(Document).where(
                Document.content_hash == content_hash,
            )
        )

        return result.scalar_one_or_none()

    async def get_all(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """
        Retrieve documents using pagination.

        Used for:
        - document dashboard
        - history views
        - admin listing
        """

        result = await self.session.execute(
            select(Document)
            .offset(offset=offset)
            .limit(limit=limit)
            .order_by(Document.created_at.desc())
        )

        return list(result.scalars().all())

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error: str | None = None,
    ) -> Document | None:
        """
        Update document lifecycle state.

        Examples:
        UPLOADED
            ↓
        PROCESSING
            ↓
        READY / FAILED
        """

        document = await self.get_by_id(document_id=document_id)
        if not document:
            return None

        document.status = status
        document.error_message = error

        await self.session.flush()
        return document

    async def update_file_path(
        self,
        document_id: str,
        file_path: str,
    ) -> Document | None:
        document = await self.get_by_id(document_id)
        if not document:
            return None

        document.file_path = file_path

        return document

    async def update_ingestion_stats(
        self,
        document_id: str,
        element_count: int,
        chunk_count: int,
    ) -> Document | None:
        """
        Store ingestion pipeline results.

        These values connect relational metadata
        with the vector database contents.
        """

        document = await self.get_by_id(document_id=document_id)
        if not document:
            return None

        document.element_count = element_count
        document.chunk_count = chunk_count

        await self.session.flush()
        return document

    async def delete(
        self,
        document_id: str,
    ) -> bool:
        """
        Delete a document from the database.

        Returns True if the document existed,
        otherwise False.
        """

        document = await self.get_by_id(document_id)
        if document is None:
            return False

        await self.session.delete(document)
        await self.session.flush()

        return True
