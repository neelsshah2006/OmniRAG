from app.core.logging import logger
from app.ai.embeddings.service import get_embedding_service
from app.storage.vector.service import get_vector_service
from app.storage.vector.models import VectorDocument
from app.rag.chunking.service import chunking_service
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

    async def ingest(self, document: Document) -> int:
        logger.info(f"Starting Ingestion: {document.id}")

        # 1. Chunk Document
        chunker = chunking_service.get_chunker(document)
        logger.info(f"Using chunker: {chunker.__class__.__name__}")
        chunks = chunker.chunk(document)
        logger.info(f"Created {len(chunks)} chunks")

        # 2. Embed Chunks

        embedding_service = get_embedding_service()
        vector_service = get_vector_service()
        vector_documents = []

        for chunk in chunks:
            if not chunk.content.strip():
                continue
            embedding = await embedding_service.embed_text(chunk.content)
            vector_documents.append(
                VectorDocument(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "element_ids": chunk.element_ids,
                        "content_length": len(chunk.content),
                        **chunk.metadata,
                    },
                )
            )

        await vector_service.add_documents(vector_documents)
        logger.success(f"Ingested document: {document.id}")

        return len(chunks)


ingestion_pipeline = IngestionPipeline()
