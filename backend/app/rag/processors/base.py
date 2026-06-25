from abc import ABC, abstractmethod

from app.rag.ingestion.models import Document


class DocumentProcessor(ABC):
    @abstractmethod
    async def process(self, document: Document) -> Document:
        pass
