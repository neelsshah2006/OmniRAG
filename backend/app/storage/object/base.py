from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from app.storage.object.models import StoredObject


class ObjectStorage(ABC):
    """
    Interface for object storage providers.

    Implementations:
    - MinIO
    - AWS S3
    - Azure Blob
    """

    @abstractmethod
    async def upload_stream(
        self,
        stream: IO[bytes],
        size: int,
        key: str,
        content_type: str | None = None,
    ) -> StoredObject:
        """
        Upload a binary stream.

        Used for:
        - FastAPI uploads
        - API requests
        """
        pass

    @abstractmethod
    async def upload_path(
        self,
        path: Path,
        key: str,
        content_type: str | None = None,
    ) -> StoredObject:
        """
        Upload an existing file.

        Used for:
        - Batch ingestion
        - CLI tools
        - Migration scripts
        """
        pass

    @abstractmethod
    async def download(
        self,
        key: str,
        destination: Path,
    ) -> Path:
        pass

    @abstractmethod
    async def delete(
        self,
        key: str,
    ) -> None:
        pass
