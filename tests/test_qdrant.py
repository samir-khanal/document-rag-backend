from app.repositories.qdrant import (
    COLLECTION_NAME,
    qdrant_client,
    create_collection,
    store_chunks,
)


def test_create_collection():
    create_collection(vector_size=384)

    collections = qdrant_client.get_collections().collections
    collection_names = {collection.name for collection in collections}

    assert COLLECTION_NAME in collection_names


def test_store_chunks():
    create_collection(vector_size=384)

    chunks = [
        "This is the first test chunk.",
        "This is the second test chunk.",
    ]

    # These are fake embeddings for testing the Qdrant repository.
    # The real ingestion pipeline will provide embeddings from
    # Sentence Transformers.
    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
    ]

    store_chunks(
        document_id="test-document-001",
        filename="test.txt",
        chunks=chunks,
        embeddings=embeddings,
        chunking_strategy="recursive",
    )

    result = qdrant_client.count(
        collection_name=COLLECTION_NAME,
    )

    assert result.count >= 2