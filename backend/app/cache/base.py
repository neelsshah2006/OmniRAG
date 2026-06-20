import json
from typing import Any

from app.core.redis import redis_manager


class CacheService:
    """
    Generic Redis cache wrapper.

    Responsibilities:
    - JSON serialization
    - TTL handling
    - Key isolation
    """

    def __init__(
        self,
        namespace: str,
        default_ttl: int = 3600,
    ):
        self.namespace = namespace
        self.default_ttl = default_ttl

    def _build_key(
        self,
        key: str,
    ) -> str:
        return f"{self.namespace}:{key}"

    async def get(
        self,
        key: str,
    ) -> Any | None:

        redis = redis_manager.get_client()

        value = await redis.get(self._build_key(key))

        if value is None:
            return None

        return json.loads(value)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:

        redis = redis_manager.get_client()

        await redis.set(
            self._build_key(key),
            json.dumps(value),
            ex=ttl or self.default_ttl,
        )

    async def delete(
        self,
        key: str,
    ) -> None:

        redis = redis_manager.get_client()

        await redis.delete(self._build_key(key))
