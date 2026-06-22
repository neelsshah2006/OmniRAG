from app.core.logging import logger
from app.ai.embeddings.service import get_embedding_service
from app.storage.vector.service import get_vector_service
from app.storage.vector.models import VectorDocument
from app.rag.chunking.base import Chunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.models import Document


class IngestionPipeline:
    """
    Converts raw documents into searchable knowledge.

    Flow:
    Document
        ↓
    Chunk
        ↓
    Embed
        ↓
    Store Vector
    """

    def __init__(self, chunker: Chunker | None = None):
        self.chunker = chunker or RecursiveChunker()

    async def ingest(self, document: Document) -> int:
        logger.info(f"Starting Ingestion: {document.id}")

        # 1. Chunk Document
        chunks = self.chunker.chunk(document=document)
        logger.info(f"Created {len(chunks)} chunks")

        # 2. Embed Chunks

        embedding_service = get_embedding_service()
        vector_service = get_vector_service()
        vector_documents = []

        for chunk in chunks:
            embedding = await embedding_service.embed_text(chunk.content)
            vector_documents.append(
                VectorDocument(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        await vector_service.add_documents(vector_documents)
        logger.success(f"Ingested document: {document.id}")

        return len(chunks)


ingestion_pipeline = IngestionPipeline()
