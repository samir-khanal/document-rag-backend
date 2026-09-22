from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "document_chunks"


client = QdrantClient(url=QDRANT_URL)


def create_collection(vector_size: int) -> None:
    """Create the document collection if it does not already exist."""

    collections = client.get_collections().collections

    existing_names = {collection.name for collection in collections}

    if COLLECTION_NAME in existing_names:
        return

    # Cosine similarity is commonly used for text embeddings because
    # it compares the direction of vectors rather than their magnitude.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )