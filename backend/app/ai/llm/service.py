from app.core.logging import logger
from app.ai.llm.base import LLMProvider
from app.ai.llm.models import (
    ChatMessage,
    GenerationConfig,
    LLMResponse,
)
from app.ai.llm.providers.groq import GroqProvider


class LLMService:
    """
    Central LLM service.

    Responsibilities:
    - Manage LLM provider lifecycle
    - Expose common generation interface
    - Hide provider implementation details
    - Future routing/fallback
    """

    def __init__(self):
        self.provider: LLMProvider | None = None

    def initialize(self) -> None:
        logger.info("Initializing LLM Service")
        self.provider = GroqProvider()
        logger.success("LLM Service Ready")

    async def generate(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> LLMResponse:
        if self.provider is None:
            raise RuntimeError("LLM Service Not Initialized")

        return await self.provider.generate(messages=messages, config=config)


llm_service = LLMService()
