from app.ai.reranker.base import RerankerProvider
from app.ai.reranker.providers.cross_encoder import CrossEncoderReranker
from app.core.logging import logger


class RerankerService:
    def __init__(self):
        self.provider: RerankerProvider | None = None

    def initialize(self):
        logger.info("Initializing Reranker")
        self.provider = CrossEncoderReranker()
        logger.success("Reranker ready")

    async def rerank(self, query: str, documents: list[str]):
        if self.provider is None:
            raise RuntimeError("Reranker not initialized")

        return await self.provider.rerank(
            query,
            documents,
        )


reranker_service = RerankerService()
