from app.rag.ingestion.models import Document

from app.rag.chunking.base import Chunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.unstructured import UnstructuredTitleChunker


class ChunkingService:
    """
    Selects best chunking strategy
    for a document.
    """

    def __init__(self):
        self.recursive = RecursiveChunker()
        self.unstructured = UnstructuredTitleChunker()

    def get_chunker(
        self,
        document: Document,
    ) -> Chunker:

        has_source_elements = any(
            element.source_element for element in document.elements
        )

        if has_source_elements:
            return self.unstructured

        return self.recursive


chunking_service = ChunkingService()
