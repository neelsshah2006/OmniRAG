import time

from groq import AsyncGroq

from app.ai.llm.base import LLMProvider
from app.ai.llm.models import ChatMessage, GenerationConfig, LLMResponse
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class GroqProvider(LLMProvider):
    """
    Groq LLM provider implementation.
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        logger.info(
            "Groq provider initialized",
            model=self.model,
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        config = config or GenerationConfig()
        try:
            logger.debug(
                "Sending request to Groq",
                model=self.model,
                messages=len(messages),
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[message.model_dump() for message in messages],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Groq response received",
                model=self.model,
                latency_ms=round(latency_ms, 2),
                tokens=response.usage.total_tokens,
            )

            return LLMResponse(
                content=(response.choices[0].message.content),
                model=self.model,
                latency_ms=round(latency_ms, 2),
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        except Exception:
            logger.exception(
                "Groq generation failed",
                model=self.model,
            )

            raise
