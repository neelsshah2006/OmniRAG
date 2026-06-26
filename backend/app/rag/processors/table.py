from app.ai.llm.models import ChatMessage, GenerationConfig
from app.ai.llm.service import llm_service
from app.ai.prompts.table import DOCUMENT_TABLE_ANALYSIS_PROMPT
from app.rag.ingestion.models import Document
from app.rag.processors.base import DocumentProcessor
from app.core.config import get_settings
from app.utils.concurrency import gather_with_limit
from app.core.logging import logger

settings = get_settings()


class TableProcessor(DocumentProcessor):
    async def process(self, document: Document) -> Document:

        table_elements = []
        for element in document.elements:
            if element.metadata.get("text_as_html") and element.content:
                table_elements.append(element)

        tasks = [
            llm_service.generate(
                messages=[
                    ChatMessage(
                        role="system",
                        content=DOCUMENT_TABLE_ANALYSIS_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=element.content,
                    ),
                ],
                config=GenerationConfig(temperature=0),
            )
            for element in table_elements
        ]

        logger.info(
            "Generating {} table summaries",
            len(table_elements),
        )

        responses = await gather_with_limit(
            tasks,
            settings.MAX_CONCURRENT_AI_REQUESTS,
        )

        for element, response in zip(table_elements, responses):
            original_content = element.content or ""
            element.content = f"""
{original_content}


[Table Analysis]

{response.content}
""".strip()

            element.metadata["table_processed"] = True

        return document
