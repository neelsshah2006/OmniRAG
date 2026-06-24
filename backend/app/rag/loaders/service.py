from pathlib import Path

from app.rag.ingestion.models import Document
from app.rag.loaders.base import DocumentLoader
from app.rag.loaders.text import TextLoader
from app.rag.loaders.unstructured import UnstructuredLoader


class LoaderService:
    """
    Selects correct loader
    based on document type.
    """

    def __init__(
        self,
    ):
        self.loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".pdf": UnstructuredLoader(),
            ".docx": UnstructuredLoader(),
            ".pptx": UnstructuredLoader(),
        }

    async def load(
        self,
        file_path: str,
    ) -> Document:

        path = Path(file_path)
        extension = path.suffix.lower()
        loader = self.loaders.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file type: {extension}")
        return await loader.load(path)


loader_service = LoaderService()
