from app.cache.base import CacheService
from app.cache.constants import EMBEDDING_CACHE_TTL
from app.cache.keys import generate_cache_key


class EmbeddingCache:
    def __init__(self):

        self.cache = CacheService(
            namespace="embedding",
            default_ttl=EMBEDDING_CACHE_TTL,
        )

    def _key(
        self,
        text: str,
        model: str,
    ) -> str:

        return generate_cache_key(
            model,
            text,
        )

    async def get_embedding(
        self,
        text: str,
        model: str,
    ) -> list[float] | None:

        key = self._key(
            text,
            model,
        )

        return await self.cache.get(key)

    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
    ) -> None:

        key = self._key(
            text,
            model,
        )

        await self.cache.set(
            key,
            embedding,
        )


embedding_cache = EmbeddingCache()
