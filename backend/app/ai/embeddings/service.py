from typing import Optional

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.embeddings.provider.sentence_transformer import (
    SentenceTransformerProvider,
)
from app.core.logging import logger


class EmbeddingService:

    def __init__(
        self,
        provider: EmbeddingProvider,
    ):
        self.provider = provider

        logger.bind(
            model=provider.model_name,
            dimension=provider.dimension,
        ).info("Embedding service initialized")

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:

        if not text.strip():
            raise ValueError("Cannot embed empty text")

        return await self.provider.embed_text(text)

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        texts = [text for text in texts if text.strip()]

        if not texts:
            return []

        return await self.provider.embed_batch(texts)

    @property
    def dimension(self):
        return self.provider.dimension

    @property
    def model_name(self):
        return self.provider.model_name


_embedding_service: Optional[EmbeddingService] = None


async def init_embedding_service():
    global _embedding_service

    logger.info("Loading embedding model...")
    provider = SentenceTransformerProvider()
    _embedding_service = EmbeddingService(provider)
    logger.success("Embedding service ready")


def get_embedding_service() -> EmbeddingService:

    if _embedding_service is None:
        raise RuntimeError("Embedding service not initialized")
    return _embedding_service
