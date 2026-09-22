from fastapi import FastAPI

from app.repositories.qdrant import qdrant_client
from app.repositories.redis import redis_client

from app.api.documents import router as documents_router


app = FastAPI(
    title="Palm Mind AI Backend",
    description="Document ingestion and conversational RAG backend",
    version="1.0.0",
)

app.include_router(documents_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/services")
def service_health_check() -> dict[str, str]:
    """
    Check the external services that the application depends on.

    This is useful during development because a working FastAPI process
    does not necessarily mean PostgreSQL, Redis, and Qdrant are reachable.
    """
    redis_client.ping()
    qdrant_client.get_collections()

    return {
        "api": "ok",
        "redis": "ok",
        "qdrant": "ok",
    }