import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    BackgroundTasks,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService
from app.storage.database.session import get_session
from app.tasks.document import process_document_task

router = APIRouter()
get_db = get_session


@router.post(
    "/upload",
    response_model=DocumentResponse,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
):
    """
    Upload a document and start asynchronous ingestion.
    """

    service = DocumentService(session=session)

    document = await service.create_document(
        filename=file.filename,
        stream=file.file,
        size=file.size,
        content_type=file.content_type,
    )

    background_tasks.add_task(
        process_document_task,
        document.id,
    )

    return document
