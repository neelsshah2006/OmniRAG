from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.document import Document, DocumentStatus
from app.models.document_event import DocumentEventType
from app.rag.ingestion.pipeline import ingestion_pipeline
from app.rag.loaders.service import loader_service
from app.repositories.document_event_repository import DocumentEventRepository
from app.repositories.document_repository import DocumentRepository


class DocumentService:
    """
    Business layer for document workflows.

    Responsibilities:
    - Manage document lifecycle
    - Coordinate ingestion pipeline
    - Maintain status/events

    Does NOT:
    - Execute database queries directly
    - Generate embeddings directly
    - Communicate with vector DB directly
    """

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session
        self.document_repo = DocumentRepository(session=session)
        self.event_repo = DocumentEventRepository(session=session)

    async def ingest_file(
        self,
        path: Path,
        content_type: str | None = None,
    ) -> Document:

        logger.info(f"Creating document workflow: {path.name}")

        async with self.session.begin():
            document = await self.document_repo.create(
                Document(
                    filename=path.name,
                    file_path=str(path),
                    file_size=path.stat().st_size,
                    content_type=content_type,
                )
            )

            await self.event_repo.create(
                document.id,
                DocumentEventType.UPLOADED,
                "Document uploaded",
            )

            await self.document_repo.update_status(
                document.id,
                DocumentStatus.PROCESSING,
            )

            await self.event_repo.create(
                document.id,
                DocumentEventType.PROCESSING_STARTED,
            )

        try:
            await self.event_repo.create(
                document.id,
                DocumentEventType.PARSING_STARTED,
                "Parsing document",
            )

            await self.session.commit()

            parsed_document = await loader_service.load(str(path))

            parsed_document.id = document.id

            chunk_count = await ingestion_pipeline.ingest(parsed_document)

            async with self.session.begin():
                await self.event_repo.create(
                    document.id,
                    DocumentEventType.PARSING_COMPLETED,
                    metadata={"elements": len(parsed_document.elements)},
                )

                await self.document_repo.update_ingestion_stats(
                    document.id,
                    element_count=len(parsed_document.elements),
                    chunk_count=chunk_count,
                )

                await self.document_repo.update_status(
                    document.id,
                    DocumentStatus.READY,
                )

                await self.event_repo.create(
                    document.id,
                    DocumentEventType.COMPLETED,
                    "Document ready for retrieval",
                    {"chunks": chunk_count},
                )

        except Exception as error:
            logger.exception("Document ingestion failed")

            async with self.session.begin():
                await self.document_repo.update_status(
                    document.id,
                    DocumentStatus.FAILED,
                    str(error),
                )

                await self.event_repo.create(
                    document.id,
                    DocumentEventType.FAILED,
                    str(error),
                )

        return await self.document_repo.get_by_id(document.id)
