from abc import ABC, abstractmethod

from app.ai.llm.models import (
    ChatMessage,
    GenerationConfig,
    LLMResponse,
)


class LLMProvider(ABC):
    """
    Base interface for all LLM providers.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        pass
