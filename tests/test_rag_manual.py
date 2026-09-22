import pytest

from app.services.rag_service import answer_question


def print_rag_result(result: dict) -> None:
    """Pretty-print the RAG answer and retrieved sources for debugging."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, src in enumerate(result["sources"], start=1):
        print(f"\nSource {i}")
        print("-" * 60)
        print(f"File:             {src['filename']}")
        print(f"Document ID:      {src['document_id']}")
        print(f"Chunk index:      {src['chunk_index']}")
        print(f"Chunking:         {src['chunking_strategy']}")
        print(f"Similarity score: {src['score']:.4f}")
        print("\nText:")
        print(src["text"])
        print("-" * 60)


# Marked as an integration test because it calls the real Gemini API.
# Excluded from the default pytest run so a quota error or Google outage
# never breaks the standard test suite. Run it on demand with:
#     pytest -m integration
@pytest.mark.integration
def test_rag_answer():
    result = answer_question(
        question="How long do I have to return a product?",
        limit=3,
    )
    print_rag_result(result)
    assert result["answer"]