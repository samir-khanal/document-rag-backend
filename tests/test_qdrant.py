from app.repositories.qdrant import (
    COLLECTION_NAME,
    client,
    create_collection,
)


def test_create_collection():
    create_collection(vector_size=384)

    collections = client.get_collections().collections
    collection_names = {collection.name for collection in collections}

    assert COLLECTION_NAME in collection_names