import uuid
from pathlib import Path

from app.rag.ingestion.models import Document, DocumentElement
from app.rag.loaders.base import DocumentLoader

MAX_FILE_SIZE_MB = 25


class TextLoader(DocumentLoader):
    """
    Loader for plain text based documents.
    """

    async def load(self, path: Path) -> Document:
        self._validate(path)
        content = path.read_text(encoding="utf-8")
        element = DocumentElement(
            id=str(uuid.uuid4()),
            type="Text",
            element_index=0,
            content=content,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
            },
        )

        return Document(
            id=str(uuid.uuid4()),
            elements=[element],
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "loader": "text",
            },
        )

    def _validate(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {size_mb:.2f} MB")
