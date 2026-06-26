from abc import ABC, abstractmethod
from pathlib import Path

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
    async def upload(
        self,
        path: Path,
        key: str,
        content_type: str | None = None,
    ) -> StoredObject:
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
