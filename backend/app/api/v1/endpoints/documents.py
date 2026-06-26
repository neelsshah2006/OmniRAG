import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService
from app.storage.database.session import get_session

router = APIRouter()
get_db = get_session


UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
):
    """
    Upload a document and start ingestion.

    Handles:
    - file persistence
    - parsing
    - chunking
    - indexing
    """

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    service = DocumentService(session=session)

    return await service.ingest_file(
        path=file_path,
        content_type=file.content_type,
    )
