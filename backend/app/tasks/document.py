from app.services.document_service import DocumentService
from app.storage.database.session import SessionLocal


async def process_document_task(
    document_id: str,
) -> None:
    """
    Background task responsible for processing
    an uploaded document.

    Uses its own database session instead of
    reusing the request-scoped session.
    """

    async with SessionLocal() as session:
        service = DocumentService(session=session)
        await service.process_document(document_id)
