from app.core.logging import logger
from app.storage.object.minio import MinIOStorage

_object_storage: MinIOStorage | None = None


def init_object_storage():
    global _object_storage

    logger.info("Initializing object storage")

    _object_storage = MinIOStorage()
    _object_storage.connect()

    logger.success("Object storage ready")


def get_object_storage() -> MinIOStorage:

    if _object_storage is None:
        raise RuntimeError("Object storage not initialized")

    return _object_storage
