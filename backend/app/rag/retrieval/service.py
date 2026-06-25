from app.core.config import get_settings
from app.core.logging import logger
from app.rag.retrieval.retrievers.base import BaseRetriever
from app.rag.retrieval.retrievers.dense import DenseRetriever
from app.rag.retrieval.retrievers.hybrid import HybridRetriever

settings = get_settings()


class RetrievalService:
    def __init__(self):
        self.retriever: BaseRetriever | None = None

    def initialize(self):
        logger.info("Initializing Retrieval Service")

        if settings.RETRIEVAL_MODE == "dense":
            self.retriever = DenseRetriever()
        elif settings.RETRIEVAL_MODE == "hybrid":
            self.retriever = HybridRetriever()
        else:
            raise ValueError(f"Unknown retrieval mode: {settings.RETRIEVAL_MODE}")

        logger.success("Retrieval Service Ready")

    async def retrieve(
        self,
        query: str,
    ):
        if self.retriever is None:
            raise RuntimeError("Retrieval service not initialized")

        return await self.retriever.retrieve(query)


retrieval_service = RetrievalService()
