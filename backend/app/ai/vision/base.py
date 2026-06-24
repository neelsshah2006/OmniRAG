from abc import ABC, abstractmethod

from app.ai.vision.models import VisionResponse


class VisionProvider(ABC):

    @abstractmethod
    async def describe_image(
        self, image_base64: str, prompt: str | None = None
    ) -> VisionResponse:
        pass
