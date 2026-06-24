from app.core.logging import logger

from app.ai.vision.base import VisionProvider
from app.ai.vision.models import VisionResponse
from app.ai.vision.providers.groq import GroqVisionProvider


class VisionService:
    """
    Central image understanding service.

    Responsibilities:
    - Manage vision provider
    - Provide common interface
    - Future routing/fallback
    """

    def __init__(self):
        self.provider: VisionProvider | None = None

    def initialize(self) -> None:
        logger.info("Initializing Vision Service")
        self.provider = GroqVisionProvider()
        logger.success("Vision Service Ready")

    async def describe_image(
        self, image_base64: str, prompt: str | None = None
    ) -> VisionResponse:

        if self.provider is None:
            raise RuntimeError("Vision Service not initialized")

        return await self.provider.describe_image(
            image_base64=image_base64,
            prompt=prompt,
        )


vision_service = VisionService()
