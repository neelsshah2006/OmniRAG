from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging, logger
from app.core.exceptions import (
    OmniRAGException,
    omnirag_exception_handler,
)

from app.middleware.request import RequestMiddleware

settings = get_settings()

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OmniRAG server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    yield
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
    logger.info("Health check requested")
    return {
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION,
    }
