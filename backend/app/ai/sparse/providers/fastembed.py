from fastembed import SparseTextEmbedding

from app.core.logging import logger
from app.ai.sparse.base import SparseEmbeddingProvider
from app.ai.sparse.models import SparseVector
from app.core.config import get_settings

settings = get_settings()


class FastEmbedProvider(SparseEmbeddingProvider):

    def __init__(self):

        logger.info("Loading sparse embedding model")
        self.model = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL)

    async def embed(self, text: str) -> SparseVector:

        embedding = list(self.model.embed([text]))[0]
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )
