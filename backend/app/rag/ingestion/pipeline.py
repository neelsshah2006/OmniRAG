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

        valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]
        if not valid_chunks:
            logger.warning(
                "No valid chunks generated for document: {}",
                document.id,
            )
            return 0

        texts = [chunk.content for chunk in valid_chunks]

        try:
            logger.info(
                "Embedding {} chunks in batch",
                len(valid_chunks),
            )

            dense_vectors, sparse_vectors = await asyncio.gather(
                embedding_service.embed_batch(texts),
                sparse_service.embed_batch(texts),
            )
            if len(dense_vectors) != len(valid_chunks) or len(sparse_vectors) != len(
                valid_chunks
            ):
                raise RuntimeError(
                    "Embedding services returned inconsistent batch sizes."
                )

        except Exception:
            logger.exception(
                "Embedding generation failed for document: {}",
                document.id,
            )
            raise

        vector_documents: list[VectorDocument] = []
        for chunk, dense_vector, sparse_vector in zip(
            valid_chunks,
            dense_vectors,
            sparse_vectors,
        ):
            vector_documents.append(
                VectorDocument(
                    id=chunk.id,
                    dense_vector=dense_vector,
                    sparse_vector=StoredSparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
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

        await vector_service.add_documents(vector_documents)
        logger.success(f"Ingested document: {document.id}")

        return len(vector_documents)


ingestion_pipeline = IngestionPipeline()
