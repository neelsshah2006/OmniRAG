from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "OmniRAG"
    API_VERSION: str = "v1"
    ENVIRONMENT: Literal[
        "development",
        "staging",
        "production",
    ] = "development"

    # Logging
    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ] = "INFO"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "omnirag_chunks"
    VECTOR_UPSERT_BATCH_SIZE: int = 256

    # Vector Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 64

    # Sparse Embeddings
    SPARSE_EMBEDDING_PROVIDER: str = "fastembed"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"

    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_CONCURRENT_AI_REQUESTS: int = 5

    # Retrieval
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_MODE: str = "dense"

    # Reranker
    RERANK_TOP_K: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MIN_RELEVANCE_SCORE: float = 0.3

    # Database URL
    DATABASE_URL: str = "sqlite+aiosqlite:///./omnirag.db"

    # Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "documents"
    MINIO_SECURE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
