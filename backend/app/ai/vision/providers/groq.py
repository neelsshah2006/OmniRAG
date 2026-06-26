from groq import (
    APIConnectionError,
    APITimeoutError,
    AsyncGroq,
    InternalServerError,
    RateLimitError,
)

from app.ai.vision.base import VisionProvider
from app.ai.vision.models import VisionResponse
from app.core.config import get_settings
from app.utils.retry import retry_async

settings = get_settings()


class GroqVisionProvider(VisionProvider):
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_VISION_MODEL

    async def describe_image(
        self,
        image_base64: str,
        prompt: str | None = None,
    ) -> VisionResponse:

        response = await retry_async(
            self.client.chat.completions.create,
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
                                "url": (f"data:image/png;base64,{image_base64}")
                            },
                        },
                    ],
                }
            ],
            retries=settings.AI_MAX_RETRIES,
            base_delay=settings.AI_RETRY_BASE_DELAY,
            retry_exceptions=(
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
            ),
            operation="Groq Vision",
        )

        return VisionResponse(
            description=response.choices[0].message.content,
            model=self.model,
        )
