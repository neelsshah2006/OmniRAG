from abc import ABC, abstractmethod

from app.rag.retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    """
    Base contract for retrieval strategies.

    Implementations:
    - Dense vector search
    - Hybrid search
    - Graph retrieval
    """

    @abstractmethod
    async def retrieve(self, query: str) -> RetrievalResult:
        pass
