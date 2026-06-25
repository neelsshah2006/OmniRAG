from app.ai.llm.models import ChatMessage, GenerationConfig
from app.ai.llm.service import llm_service
from app.ai.prompts.query import QUERY_REWRITE_PROMPT


class QueryRewriter:
    async def rewrite(self, query: str) -> str:
        response = await llm_service.generate(
            messages=[
                ChatMessage(
                    role="system",
                    content=QUERY_REWRITE_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=query,
                ),
            ],
            config=GenerationConfig(
                temperature=0,
            ),
        )

        return response.content.strip()


query_rewriter = QueryRewriter()
