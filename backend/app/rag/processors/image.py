from app.core.logging import logger
from app.rag.processors.base import DocumentProcessor
from app.rag.ingestion.models import Document
from app.ai.vision.service import vision_service
from app.ai.prompts.vision import DOCUMENT_IMAGE_ANALYSIS_PROMPT


class ImageProcessor(DocumentProcessor):
    """
    Converts image elements into
    searchable text.
    """

    async def process(self, document: Document) -> Document:

        for element in document.elements:
            image_base64 = element.metadata.get("image_base64")
            if not image_base64:
                continue

            logger.info("Generating image summary")

            response = await vision_service.describe_image(
                image_base64=image_base64,
                prompt=DOCUMENT_IMAGE_ANALYSIS_PROMPT,
            )

            logger.info("Image Summary generated")

            original_content = element.content or ""
            element.content = f"""
{original_content}

[Image Analysis]

{response.description}
""".strip()
            element.metadata["vision_model"] = response.model

        return document
