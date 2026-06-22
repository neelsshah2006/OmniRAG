from app.rag.retrieval.models import RetrievalResult


class ContextBuilder:
    """
    Converts retrieved chunks into
    LLM-ready context.
    """

    def build(self, result: RetrievalResult) -> str:
        if not result.chunks:
            return "No relevant context found"

        context_parts = []
        for index, chunk in enumerate(result.chunks, start=1):
            source = chunk.metadata.get("filename", "unknown")
            context_parts.append(
                f"\n[Source {index}]\nFile: {source}\nRelevance Score: {chunk.score}\n\n{chunk.content}"
            )

        return "\n".join(context_parts)


context_builder = ContextBuilder()
