import asyncio

from sentence_transformers import SentenceTransformer

from app.ai.embeddings.base import EmbeddingProvider
from app.core.config import get_settings

settings = get_settings()


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self._model_name = model_name

        self._model = SentenceTransformer(model_name)

        self._dimension = self._model.get_embedding_dimension()

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:

        embeddings = await asyncio.to_thread(
            self._model.encode,
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single model call.
        """

        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
