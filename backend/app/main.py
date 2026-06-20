from contextlib import asynccontextmanager
from fastapi import FastAPI
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.logging import setup_logging, logger
from app.core.exceptions import (
    OmniRAGException,
    omnirag_exception_handler,
)
from app.core.redis import redis_manager

from app.middleware.request import RequestMiddleware

settings = get_settings()

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OmniRAG server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Connect infrastructure services
    await redis_manager.connect()
    logger.info("Redis connected")

    yield

    # Shutdown services gracefully
    await redis_manager.close()
    logger.info("Redis disconnected")

    logger.info("Shutting down OmniRAG server")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_exception_handler(
    OmniRAGException,
    omnirag_exception_handler,
)

app.add_middleware(RequestMiddleware)


@app.get("/health")
async def health():

    redis_status = "healthy"

    try:
        redis = redis_manager.get_client()
        await redis.ping()

    except RedisError:
        redis_status = "unhealthy"

    return {
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION,
        "services": {
            "redis": redis_status,
        },
    }
