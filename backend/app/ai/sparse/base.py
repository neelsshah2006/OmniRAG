from abc import ABC, abstractmethod

from app.ai.sparse.models import SparseVector


class SparseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> SparseVector:
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        pass
