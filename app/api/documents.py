from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.repositories.postgres import get_db, save_document
from app.repositories.qdrant import store_chunks
from app.services.chunking import ChunkingStrategy, chunk_document
from app.services.document_service import extract_text
from app.services.embedding_service import embed_documents

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
    db: Session = Depends(get_db),
) -> dict:
    """Upload, process, embed, and store a document."""

    try:
        text = await extract_text(file)

        chunks = chunk_document(
            text=text,
            strategy=chunking_strategy,
        )

        if not chunks:
            raise ValueError("Document does not contain usable text.")

        # The same embedding model must be used for both document chunks
        # and user queries later during RAG retrieval.
        embeddings = embed_documents(chunks)

        # Generate the document ID once and reuse it across PostgreSQL and
        # Qdrant. PostgreSQL stores it as a UUID, while Qdrant stores the
        # same identifier as a string in its payload.
        document_id = uuid4()

        store_chunks(
            document_id=str(document_id),
            filename=file.filename or "unknown",
            chunks=chunks,
            embeddings=embeddings,
            chunking_strategy=chunking_strategy.value,
        )

        # Qdrant stores the searchable chunks and embeddings, while
        # PostgreSQL stores structured metadata about the original document.
        # Keeping these responsibilities separate makes each system useful
        # for the type of data it is designed to handle.
        save_document(
            db=db,
            document_id=document_id,
            filename=file.filename or "unknown",
            file_type=file.content_type or "unknown",
            chunking_strategy=chunking_strategy.value,
            chunk_count=len(chunks),
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
        "document_id": str(document_id),
        "filename": file.filename or "unknown",
        "file_type": file.content_type or "unknown",
        "chunking_strategy": chunking_strategy.value,
        "text_length": len(text),
        "chunk_count": len(chunks),
    }