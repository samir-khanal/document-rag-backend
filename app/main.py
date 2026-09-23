from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.repositories.qdrant import create_collection, qdrant_client
from app.repositories.redis import redis_client
from app.services.embedding_service import EMBEDDING_DIM

from app.api.documents import router as documents_router
from app.api.chat import router as chat_router 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize infrastructure before the API accepts requests.

    Why this lives here and not at module level in qdrant.py:
    importing a repository module should not have side effects. Startup
    is the correct place for one-time setup. The create_collection call
    is idempotent — it creates the collection only if it doesn't exist,
    so this is safe to run on every startup.
    """
    create_collection(vector_size=EMBEDDING_DIM)
    yield

app = FastAPI(
    title="Palm Mind AI Backend",
    description="Document ingestion and conversational RAG backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(chat_router) 

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