from abc import ABC, abstractmethod

from app.ai.sparse.models import SparseVector


class SparseEmbeddingProvider(ABC):

    @abstractmethod
    async def embed(self, text: str) -> SparseVector:
        pass
