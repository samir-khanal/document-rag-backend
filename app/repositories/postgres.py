from collections.abc import Generator
from datetime import datetime
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.document import Document
from app.models.booking import Booking

settings = get_settings()

engine = create_engine(
    settings.postgres_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Provide one database session per request and make sure it is closed
    afterward, even if the request fails.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_document(
    db: Session,
    document_id: UUID,
    filename: str,
    file_type: str,
    chunking_strategy: str,
    chunk_count: int,
) -> Document:
    """Save document metadata to PostgreSQL."""

    document = Document(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        chunking_strategy=chunking_strategy,
        chunk_count=chunk_count,
        created_at=datetime.now().astimezone(),
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document

def save_booking(
    db: Session,
    booking_id: UUID,
    name: str,
    email: str,
    interview_date,
    interview_time,
) -> Booking:
    """
    Persist a validated interview booking to PostgreSQL.

    The caller is responsible for validating name/email/date/time
    before this function is called. This function only writes.
    """

    booking = Booking(
        booking_id=booking_id,
        name=name,
        email=email,
        interview_date=interview_date,
        interview_time=interview_time,
        created_at=datetime.now().astimezone(),
    )

    db.add(booking)
    db.commit()
    db.refresh(booking)

    return booking