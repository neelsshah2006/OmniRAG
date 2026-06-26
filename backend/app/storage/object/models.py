from pydantic import BaseModel


class StoredObject(BaseModel):
    bucket: str
    key: str
    size: int
    content_type: str | None = None
