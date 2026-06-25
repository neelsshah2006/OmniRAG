import asyncio

from app.ai.embeddings.service import get_embedding_service
from app.ai.sparse.service import sparse_embedding_service
from app.core.logging import logger
from app.rag.chunking.service import chunking_service
from app.rag.ingestion.models import Document
from app.rag.processors.service import processor_pipeline
from app.storage.vector.models import StoredSparseVector, VectorDocument
from app.storage.vector.service import get_vector_service


class IngestionPipeline:
    """
    Converts parsed documents into searchable knowledge.

    Flow:

    Document
        ↓
    Process / Enrich
        ↓
    Chunk
        ↓
    Embed
        ↓
    Store Vector
    """

    async def ingest(self, document: Document) -> int:
        logger.info(f"Starting Ingestion: {document.id}")

        # 1. Process Document
        document = await processor_pipeline.process(document)

        # 2. Chunk Document
        chunker = chunking_service.get_chunker(document)
        logger.info(f"Using chunker: {chunker.__class__.__name__}")
        chunks = chunker.chunk(document)
        logger.info(f"Created {len(chunks)} chunks")

        # 3. Generate retrieval representations
        embedding_service = get_embedding_service()
        sparse_service = sparse_embedding_service
        vector_service = get_vector_service()
        vector_documents = []

        for chunk in chunks:
            if not chunk.content.strip():
                continue

            try:
                dense_embedding, sparse_embedding = await asyncio.gather(
                    embedding_service.embed_text(chunk.content),
                    sparse_service.embed(chunk.content),
                )
            except Exception as e:
                logger.error(f"Failed embedding chunk {chunk.id}: {e}")
                continue

            vector_documents.append(
                VectorDocument(
                    id=chunk.id,
                    dense_vector=dense_embedding,
                    sparse_vector=StoredSparseVector(
                        indices=sparse_embedding.indices,
                        values=sparse_embedding.values,
                    ),
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

        if not vector_documents:
            logger.warning(f"No valid chunks generated for document: {document.id}")
            return 0

        await vector_service.add_documents(vector_documents)
        logger.success(f"Ingested document: {document.id}")

        return len(vector_documents)


ingestion_pipeline = IngestionPipeline()
