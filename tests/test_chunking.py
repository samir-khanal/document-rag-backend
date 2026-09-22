from app.services.chunking import (
    ChunkingStrategy,
    chunk_document,
    fixed_chunking,
    recursive_chunking,
)

def test_fixed_chunking():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = fixed_chunking(
        text,
        chunk_size=10,
        overlap=2,
    )

    print(chunks)

    assert len(chunks) > 0
    assert chunks[0] == "ABCDEFGHIJ"


def test_recursive_chunking():
    text = (
        "This is the first sentence. "
        "This is the second sentence.\n\n"
        "This is a new paragraph. "
        "This is another sentence in that paragraph."
    )

    chunks = recursive_chunking(
        text,
        chunk_size=50,
        overlap=10,
    )

    print(chunks)

    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)

def test_chunk_document_with_fixed_strategy():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = chunk_document(
        text=text,
        strategy=ChunkingStrategy.FIXED,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) > 0
    assert chunks[0] == "ABCDEFGHIJ"


def test_chunk_document_with_recursive_strategy():
    text = (
        "This is the first sentence. "
        "This is the second sentence.\n\n"
        "This is a new paragraph. "
        "This is another sentence in that paragraph."
    )

    chunks = chunk_document(
        text=text,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=50,
        overlap=10,
    )

    assert len(chunks) > 0