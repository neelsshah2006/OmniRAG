from abc import ABC, abstractmethod

from app.rag.ingestion.models import (
    Document,
    DocumentChunk,
)


class Chunker(ABC):
    """
    Base interface for all chunking strategies.

    Examples:
    - Recursive chunking
    - Semantic chunking
    - Agentic chunking
    """

    @abstractmethod
    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        pass
