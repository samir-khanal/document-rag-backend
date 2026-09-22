from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model once when the service starts.
# Re-loading it for every document would waste time and memory.
model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text."""

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding = model.encode(text)

    return embedding.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple text chunks."""

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Document chunks cannot be empty.")

    embeddings = model.encode(texts)

    return embeddings.tolist()