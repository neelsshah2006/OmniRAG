import asyncio

from sentence_transformers import SentenceTransformer

from app.ai.embeddings.base import EmbeddingProvider


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
        )

        return embeddings.tolist()

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
