from dotenv import load_dotenv

load_dotenv()

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
from app.ai.embeddings.service import (
    init_embedding_service,
    get_embedding_service,
)
from app.storage.vector.service import (
    init_vector_service,
    close_vector_service,
)
from app.ai.llm.service import llm_service
from app.ai.vision.service import vision_service
from app.rag.retrieval.service import retrieval_service
from app.ai.sparse.service import sparse_embedding_service
from app.ai.reranker.service import reranker_service

from app.middleware.request import RequestMiddleware

settings = get_settings()

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OmniRAG server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Infrastructure
    await redis_manager.connect()
    logger.info("Redis connected")

    # AI services
    await init_embedding_service()
    embedding_service = get_embedding_service()
    logger.info("Embedding service loaded")

    # Sparse embeddings
    sparse_embedding_service.initialize()
    logger.info("Sparse embedding service loaded")

    # Vector database
    await init_vector_service(dimension=embedding_service.dimension)
    logger.info("Vector Service loaded")

    # Retrieval Service
    retrieval_service.initialize()
    logger.info("Retrieval service loaded")

    # Reranker Service
    reranker_service.initialize()
    logger.info("Reranker service loaded")

    # LLM service
    llm_service.initialize()
    logger.info("LLM Service loaded")

    # Vision Service
    vision_service.initialize()
    logger.info("Vision Service loaded")

    logger.info("OmniRAG services ready")

    yield

    # Shutdown
    await close_vector_service()
    logger.info("Qdrant Closed")

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
