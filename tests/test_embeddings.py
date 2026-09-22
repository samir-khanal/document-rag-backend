from app.services.embedding_service import (
    embed_documents,
    embed_text,
)


def test_embed_text():
    text = "This is a test document."

    embedding = embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)


def test_embed_documents():
    texts = [
        "This is the first document.",
        "This is the second document.",
    ]

    embeddings = embed_documents(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert all(len(embedding) > 0 for embedding in embeddings)


def test_same_text_produces_same_embedding():
    text = "Palm Mind AI RAG system."

    first = embed_text(text)
    second = embed_text(text)

    assert first == second