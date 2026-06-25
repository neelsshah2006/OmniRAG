from abc import ABC, abstractmethod

from app.ai.reranker.models import RerankResult


class RerankerProvider(ABC):

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        pass
