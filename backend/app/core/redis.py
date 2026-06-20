from redis.asyncio import Redis

from app.core.config import get_settings

settings = get_settings()


class RedisManager:
    """
    Central Redis connection manager.

    Responsibilities:
    - Create Redis connection
    - Maintain single client instance
    - Close connection gracefully
    """

    def __init__(self):
        self.client: Redis | None = None

    async def connect(self) -> None:
        self.client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )

        await self.client.ping()

    async def close(self) -> None:
        if self.client:
            await self.client.close()

    def get_client(self) -> Redis:
        if self.client is None:
            raise RuntimeError("Redis client is not initialized")

        return self.client


redis_manager = RedisManager()
