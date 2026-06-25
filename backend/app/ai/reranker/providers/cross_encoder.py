from sentence_transformers import CrossEncoder

from app.ai.reranker.base import RerankerProvider
from app.ai.reranker.models import RerankResult
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class CrossEncoderReranker(RerankerProvider):
    def __init__(self):

        self.model_name = settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        results = []
        for index, score in enumerate(scores):
            results.append(
                RerankResult(
                    index=index,
                    score=float(score),
                )
            )

        return sorted(
            results,
            key=lambda x: x.score,
            reverse=True,
        )
