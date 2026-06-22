import uuid
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.rag.chunking.base import Chunker

from app.rag.ingestion.models import (
    Document,
    DocumentChunk,
)


class RecursiveChunker(Chunker):

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        texts = self.splitter.split_text(document.content)

        chunks = []

        for index, text in enumerate(texts):

            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    (document.id + str(index) + text),
                )
            )

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=text,
                    chunk_index=index,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "recursive",
                    },
                )
            )

        return chunks
