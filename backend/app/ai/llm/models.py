from typing import Any
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):

    temperature: float = 0.2
    max_tokens: int = 1024


class ChatMessage(BaseModel):
    """
    Generic chat message.
    """

    role: str
    content: str


class LLMResponse(BaseModel):
    """
    Standard response returned by any LLM.
    """

    content: str
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
