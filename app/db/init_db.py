from sqlalchemy import create_engine

from app.core.config import get_settings
from app.db.base import Base
from app.models.document import Document
from app.models.booking import Booking 


settings = get_settings()

engine = create_engine(
    settings.postgres_url,
    pool_pre_ping=True,
)


def init_database() -> None:
    """Create database tables that do not already exist."""

    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_database()
    print("Database tables initialized.")