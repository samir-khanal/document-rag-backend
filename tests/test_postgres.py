from uuid import uuid4

from app.models.document import Document
from app.repositories.postgres import SessionLocal, save_document


def test_save_document():
    """Test that document metadata is saved and can be retrieved."""

    db = SessionLocal()
    document_id = uuid4()

    try:
        saved_document = save_document(
            db=db,
            document_id=document_id,
            filename="test_document.pdf",
            file_type="application/pdf",
            chunking_strategy="recursive",
            chunk_count=3,
        )

        retrieved_document = db.get(Document, document_id)

        assert retrieved_document is not None
        assert retrieved_document.document_id == document_id
        assert retrieved_document.filename == "test_document.pdf"
        assert retrieved_document.file_type == "application/pdf"
        assert retrieved_document.chunking_strategy == "recursive"
        assert retrieved_document.chunk_count == 3
        assert saved_document.document_id == document_id

    finally:
        # Remove the test record so repeated test runs do not
        # leave unnecessary data in the development database.
        document = db.get(Document, document_id)

        if document:
            db.delete(document)
            db.commit()

        db.close()