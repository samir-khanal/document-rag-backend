from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.chunking import ChunkingStrategy, chunk_document
from app.services.document_service import extract_text


router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
) -> dict:
    """Upload a document, extract its text, and split it into chunks."""

    try:
        text = await extract_text(file)

        chunks = chunk_document(
            text=text,
            strategy=chunking_strategy,
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="TXT file must use UTF-8 encoding.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "filename": file.filename or "unknown",
        "file_type": file.content_type or "unknown",
        "chunking_strategy": chunking_strategy.value,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }