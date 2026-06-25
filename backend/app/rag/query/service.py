from app.core.logging import logger
from app.rag.query.rewriter import query_rewriter


class QueryService:

    async def process(
        self,
        query: str,
    ) -> str:

        logger.info("Processing query")
        rewritten = await query_rewriter.rewrite(query)
        if rewritten != query:
            logger.info(f"Query rewritten: {rewritten}")
        return rewritten


query_service = QueryService()
