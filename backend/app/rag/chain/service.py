import time

from app.core.logging import logger
from app.ai.llm.models import (
    ChatMessage,
    GenerationConfig,
)
from app.ai.llm.service import llm_service
from app.rag.retrieval.retriever import retriever
from app.rag.chain.models import (
    RAGResponse,
    SourceReference,
)
from app.rag.retrieval.context import context_builder
from app.ai.prompts.rag import RAG_SYSTEM_PROMPT, build_rag_user_prompt


class RAGChain:
    """
    Complete Retrieval Augmented Generation pipeline.

    Responsibilities:
    - Retrieve relevant knowledge
    - Build grounded prompt
    - Generate answer
    - Attach citations
    """

    async def ask(self, question: str) -> RAGResponse:
        start_time = time.perf_counter()
        logger.info("Starting RAG Chain")
        retrieval_result = await retriever.retrieve(query=question)
        context = context_builder.build(retrieval_result)
        messages = [
            ChatMessage(
                role="system",
                content=(RAG_SYSTEM_PROMPT),
            ),
            ChatMessage(
                role="user",
                content=build_rag_user_prompt(context=context, question=question),
            ),
        ]

        llm_response = await llm_service.generate(
            messages=messages,
            config=GenerationConfig(temperature=0.2),
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=llm_response.content,
            sources=[
                SourceReference(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=chunk.score,
                )
                for chunk in retrieval_result.chunks
            ],
            model=llm_response.model,
            usage=llm_response.usage,
            latency_ms=round(latency_ms, 2),
        )


rag_chain = RAGChain()
