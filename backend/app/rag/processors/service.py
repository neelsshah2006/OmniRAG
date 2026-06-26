import asyncio

from app.core.logging import logger
from app.rag.ingestion.models import Document
from app.rag.processors.base import DocumentProcessor
from app.rag.processors.image import ImageProcessor
from app.rag.processors.table import TableProcessor


class ProcessorPipeline:
    """
    Runs document enhancement steps before chunking.

    Examples:
    - image understanding
    - OCR enhancement
    - table enrichment
    """

    def __init__(self):
        self.processors: list[DocumentProcessor] = [
            ImageProcessor(),
            TableProcessor(),
        ]

    async def process(self, document: Document) -> Document:

        logger.info(
            "Running {} document processors",
            len(self.processors),
        )

        tasks = []

        for processor in self.processors:
            logger.info(
                "Scheduling processor: {}",
                processor.__class__.__name__,
            )
            tasks.append(processor.process(document))

        await asyncio.gather(*tasks)

        return document


processor_pipeline = ProcessorPipeline()
