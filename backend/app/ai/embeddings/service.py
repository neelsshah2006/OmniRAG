from typing import Optional

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.embeddings.provider.sentence_transformer import SentenceTransformerProvider
from app.cache.embedding import embedding_cache
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

        cached_embedding = await embedding_cache.get_embedding(
            text=text,
            model=self.model_name,
        )

        if cached_embedding is not None:
            logger.debug("Embedding cache hit")
            return cached_embedding

        logger.debug("Embedding cache miss")
        embedding = await self.provider.embed_text(text)
        await embedding_cache.set_embedding(
            text=text,
            model=self.model_name,
            embedding=embedding,
        )

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)

        missing_indices = []
        missing_texts = []

        for i, text in enumerate(texts):

            if not text.strip():
                raise ValueError("Cannot embed empty text")

            cached = await embedding_cache.get_embedding(
                text=text,
                model=self.model_name,
            )

            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        logger.debug(
            "Embedding batch: {} texts ({} hits, {} misses)",
            len(texts),
            len(texts) - len(missing_texts),
            len(missing_texts),
        )

        new_vectors = await self.provider.embed_batch(missing_texts)
        for index, text, vector in zip(
            missing_indices,
            missing_texts,
            new_vectors,
        ):
            await embedding_cache.set_embedding(
                text=text,
                model=self.model_name,
                embedding=vector,
            )
            results[index] = vector

        assert all(result is not None for result in results)

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
