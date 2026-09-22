from app.services.llm_service import generate_answer
from app.services.retrieval_service import retrieve_relevant_chunks


def build_context(chunks: list[dict]) -> str:
    """Combine retrieved chunks into context for the language model."""

    if not chunks:
        return ""

    context_parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source {index}: {filename}]\n{text}"
        )

    return "\n\n".join(context_parts)


def answer_question(
    question: str,
    limit: int = 5,
    history: list[dict] | None = None,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Retrieval and generation are kept as separate steps so we can
    # inspect or test retrieval independently from the language model.
    retrieved_chunks = retrieve_relevant_chunks(
        query=question,
        limit=limit,
    )

    context = build_context(retrieved_chunks)

    if not context:
        return {
            "answer": "I could not find relevant information in the provided documents.",
            "sources": [],
        }

    # History is passed only to the LLM, not to retrieval. This keeps
    # retrieval predictable — the same question always returns the
    # same chunks regardless of what was asked earlier.
    answer = generate_answer(
        question=question,
        context=context,
        history=history,
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }