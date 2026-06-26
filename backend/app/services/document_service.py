from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.document import Document, DocumentStatus
from app.models.document_event import DocumentEventType
from app.rag.ingestion.pipeline import ingestion_pipeline
from app.rag.loaders.service import loader_service
from app.repositories.document_event_repository import DocumentEventRepository
from app.repositories.document_repository import DocumentRepository
from app.storage.object.service import get_object_storage


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

    @staticmethod
    def _object_key(
        document_id: str,
        filename: str,
    ) -> str:
        return f"documents/{document_id}/{filename}"

    async def create_document(
        self,
        *,
        filename: str,
        stream: IO[bytes],
        size: int,
        content_type: str | None = None,
    ) -> Document:
        """
        Registers a newly uploaded document.

        Responsibilities:
        - Create the document metadata record
        - Upload the original file to object storage (MinIO)
        - Store the object storage reference
        - Record upload lifecycle events

        Does NOT:
        - Parse the document
        - Generate chunks or embeddings
        - Index content into the vector database

        Processing is performed separately by `process_document()`,
        allowing asynchronous/background ingestion and document reprocessing.
        """

        logger.info(
            "Creating document: {}",
            filename,
        )

        async with self.session.begin():
            document = await self.document_repo.create(
                Document(
                    filename=filename,
                    file_size=size,
                    content_type=content_type,
                )
            )

            await self.event_repo.create(
                document.id,
                DocumentEventType.UPLOADED,
                "Document uploaded",
            )

        stored_object = None
        object_storage = get_object_storage()
        object_key = self._object_key(
            document_id=document.id,
            filename=filename,
        )

        try:
            stored_object = await object_storage.upload_stream(
                stream=stream,
                size=size,
                key=object_key,
                content_type=content_type,
            )

            async with self.session.begin():
                await self.document_repo.update_file_path(
                    document.id,
                    stored_object.key,
                )

        except Exception:
            if stored_object is not None:
                await object_storage.delete(
                    stored_object.key,
                )

            async with self.session.begin():
                await self.document_repo.delete(document.id)

            raise

        return document

    async def process_document(
        self,
        document_id: str,
    ) -> None:
        """
        Processes an uploaded document into searchable knowledge.

        Responsibilities:
        - Download the original file from object storage
        - Parse and enrich the document
        - Generate chunks and embeddings
        - Store searchable vectors
        - Update processing status and events

        This method can be executed synchronously or
        scheduled as a background task.
        """

        document = await self.document_repo.get_by_id(document_id)
        if document is None:
            raise ValueError(f"Document '{document_id}' not found.")

        object_storage = get_object_storage()
        logger.info(f"Processing document: {document.id}")

        try:
            async with self.session.begin():
                await self.document_repo.update_status(
                    document.id,
                    DocumentStatus.PROCESSING,
                )

                await self.event_repo.create(
                    document.id,
                    DocumentEventType.PROCESSING_STARTED,
                )

                await self.event_repo.create(
                    document.id,
                    DocumentEventType.PARSING_STARTED,
                    "Parsing document",
                )

            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / document.filename

                await object_storage.download(
                    key=document.file_path,
                    destination=temp_path,
                )

                parsed_document = await loader_service.load(str(temp_path))

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

            logger.success(f"Processed document: {document.id}")

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
