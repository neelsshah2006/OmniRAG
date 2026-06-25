from app.ai.sparse.providers.fastembed import FastEmbedProvider
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class SparseEmbeddingService:
    def __init__(self):

        self.provider = None

    def initialize(self):

        logger.info("Initializing sparse embeddings")

        if settings.SPARSE_EMBEDDING_PROVIDER == "fastembed":
            self.provider = FastEmbedProvider()
        else:
            raise ValueError("Unknown sparse provider")

        logger.success("Sparse embeddings ready")

    async def embed(self, text: str):

        if self.provider is None:
            raise RuntimeError("Sparse embeddings not initialized")

        return await self.provider.embed(text)


sparse_embedding_service = SparseEmbeddingService()
