from abc import ABC, abstractmethod
from pathlib import Path

from app.rag.ingestion.models import Document


class DocumentLoader(ABC):
    """
    Base interface for all document loaders.

    Converts external sources
    into internal Document objects.
    """

    @abstractmethod
    async def load(
        self,
        path: Path,
    ) -> Document:
        pass
