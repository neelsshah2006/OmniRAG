import uuid

from unstructured.chunking.title import chunk_by_title

from app.rag.chunking.base import Chunker

from app.rag.ingestion.models import (
    Document,
    DocumentChunk,
)


class UnstructuredTitleChunker(Chunker):
    """
    Uses Unstructured's native
    structure-aware chunking.
    """

    def __init__(
        self,
        max_characters: int = 3000,
        new_after_n_chars: int = 2000,
        combine_text_under_n_chars: int = 500,
        overlap: int = 200,
    ):
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        source_to_id = {}
        for element in document.elements:
            if element.source_element:
                source_to_id[id(element.source_element)] = element.id

        raw_elements = [
            element.source_element
            for element in document.elements
            if element.source_element
        ]

        if not raw_elements:
            return []

        raw_chunks = chunk_by_title(
            raw_elements,
            max_characters=self.max_characters,
            new_after_n_chars=(self.new_after_n_chars),
            combine_text_under_n_chars=(self.combine_text_under_n_chars),
            overlap=self.overlap,
            overlap_all=False,
            include_orig_elements=True,
            multipage_sections=True,
            isolate_table=True,
            skip_table_chunking=True,
            repeat_table_headers=True,
        )

        chunks = []

        for index, chunk in enumerate(raw_chunks):
            orig_elements = chunk.metadata.orig_elements or []
            element_ids = [
                source_to_id[id(source)]
                for source in orig_elements
                if id(source) in source_to_id
            ]

            chunks.append(
                DocumentChunk(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS, document.id + str(index) + chunk.text
                        )
                    ),
                    document_id=document.id,
                    content=chunk.text,
                    chunk_index=index,
                    element_ids=element_ids,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "unstructured_title",
                        "element_count": len(element_ids),
                        "category": chunk.category,
                    },
                )
            )

        return chunks
