from app.ai.prompts.vision import DOCUMENT_IMAGE_ANALYSIS_PROMPT
from app.ai.vision.service import vision_service
from app.core.logging import logger
from app.rag.ingestion.models import Document
from app.rag.processors.base import DocumentProcessor
from app.utils.concurrency import gather_with_limit
from app.core.config import get_settings

settings = get_settings()


class ImageProcessor(DocumentProcessor):
    """
    Converts image elements into
    searchable text.
    """

    async def process(self, document: Document) -> Document:

        image_elements = []

        for element in document.elements:
            image = element.metadata.get("image_base64")
            if image:
                image_elements.append(element)

        tasks = [
            vision_service.describe_image(
                image_base64=element.metadata.get("image_base64"),
                prompt=DOCUMENT_IMAGE_ANALYSIS_PROMPT,
            )
            for element in image_elements
        ]

        logger.info(
            "Generating {} image summaries",
            len(image_elements),
        )

        responses = await gather_with_limit(
            tasks,
            settings.MAX_CONCURRENT_AI_REQUESTS,
        )

        for element, response in zip(
            image_elements,
            responses,
        ):
            logger.info("Image Summary generated")

            original_content = element.content or ""
            element.content = f"""
{original_content}

[Image Analysis]

{response.description}
""".strip()
            element.metadata["vision_model"] = response.model

        return document
