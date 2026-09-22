from app.repositories.qdrant import search_chunks
from app.services.embedding_service import embed_text


def retrieve_relevant_chunks(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Embed a user query and retrieve the most relevant document chunks."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # The query must be embedded with the same Sentence Transformer model
    # used to create the document embeddings during ingestion.
    query_embedding = embed_text(query)

    results = search_chunks(
        query_embedding=query_embedding,
        limit=limit,
    )

    # Guard against duplicates — same chunk coming back twice
    # (happens if a document was ingested more than once).
    seen = set()
    unique: list[dict] = []

    for chunk in results:
        key = (chunk["filename"], chunk["chunk_index"])
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique