from pathlib import Path
import uuid
from unstructured.partition.auto import partition

from app.core.logging import logger
from app.rag.loaders.base import DocumentLoader
from app.rag.ingestion.models import (
    Document,
    DocumentElement,
)

MAX_FILE_SIZE_MB = 100


class UnstructuredLoader(DocumentLoader):
    """
    High quality document parser.

    Supports:
    - PDF
    - DOCX
    - PPTX

    Preserves:
    - layout
    - tables
    - page metadata
    """

    async def load(self, path: Path) -> Document:
        self._validate(path)
        logger.info(f"Parsing document using Unstructured: {path.name}")

        raw_elements = partition(
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )

        elements = []

        for index, element in enumerate(raw_elements):
            metadata = element.metadata.to_dict() if element.metadata else {}
            content = element.text

            if element.category == "Table":
                html = metadata.get("text_as_html")
                if html:
                    content = (
                        "Table HTML:\n" + html + "\n\nTable Text:\n" + element.text
                    )
            elements.append(
                DocumentElement(
                    id=str(uuid.uuid4()),
                    type=element.category,
                    element_index=index,
                    content=content,
                    metadata=metadata,
                    source_element=element,
                )
            )

        logger.info(f"Extracted {len(elements)} elements")

        return Document(
            id=str(uuid.uuid4()),
            elements=elements,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "loader": "unstructured",
            },
        )

    def _validate(
        self,
        path: Path,
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {size_mb:.2f} MB")
