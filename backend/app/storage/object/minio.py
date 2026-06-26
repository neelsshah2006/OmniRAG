from pathlib import Path

from minio import Minio

from app.core.config import get_settings
from app.core.logging import logger
from app.storage.object.base import ObjectStorage
from app.storage.object.models import StoredObject

settings = get_settings()


class MinIOStorage(ObjectStorage):
    def __init__(self):
        self.client: Minio | None = None

    def connect(self):

        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

        if not self.client.bucket_exists(settings.MINIO_BUCKET):
            self.client.make_bucket(settings.MINIO_BUCKET)

        logger.info("MinIO connected")

    async def upload(
        self,
        path: Path,
        key: str,
        content_type: str | None = None,
    ) -> StoredObject:

        self.client.fput_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=key,
            file_path=str(path),
            content_type=content_type,
        )

        return StoredObject(
            bucket=settings.MINIO_BUCKET,
            key=key,
            size=path.stat().st_size,
            content_type=content_type,
        )

    async def download(
        self,
        key: str,
        destination: Path,
    ) -> Path:

        self.client.fget_object(
            settings.MINIO_BUCKET,
            key,
            str(destination),
        )

        return destination

    async def delete(
        self,
        key: str,
    ) -> None:

        self.client.remove_object(
            settings.MINIO_BUCKET,
            key,
        )
