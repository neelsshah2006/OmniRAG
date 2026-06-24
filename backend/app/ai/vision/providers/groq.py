from groq import AsyncGroq

from app.core.config import get_settings
from app.core.logging import logger

from app.ai.vision.base import VisionProvider
from app.ai.vision.models import VisionResponse

settings = get_settings()


class GroqVisionProvider(VisionProvider):

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_VISION_MODEL

    async def describe_image(self, image_base64: str, prompt: str | None = None):

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (prompt or "Describe this image for retrieval."),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )

        return VisionResponse(
            description=response.choices[0].message.content,
            model=self.model,
        )
