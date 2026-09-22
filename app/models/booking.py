from datetime import date, datetime, time
from uuid import UUID

from sqlalchemy import Date, DateTime, String, Time, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Booking(Base):
    """
    Persisted interview booking.

    Bookings live in PostgreSQL (not Redis) because they are business
    records — they must survive Redis restarts and cache wipes.
    """

    __tablename__ = "bookings"

    booking_id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    interview_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    interview_time: Mapped[time] = mapped_column(
        Time,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )