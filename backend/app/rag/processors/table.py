from app.rag.processors.base import DocumentProcessor
from app.rag.ingestion.models import Document
from app.ai.llm.models import ChatMessage, GenerationConfig
from app.ai.llm.service import llm_service
from app.ai.prompts.table import DOCUMENT_TABLE_ANALYSIS_PROMPT


class TableProcessor(DocumentProcessor):

    async def process(self, document: Document) -> Document:

        for element in document.elements:

            table_html = element.metadata.get("text_as_html")
            if not table_html:
                continue

            if not element.content:
                continue

            response = await llm_service.generate(
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

            original_content = element.content or ""
            element.content = f"""
{original_content}


[Table Analysis]

{response.content}
""".strip()

            element.metadata["table_processed"] = True

        return document
