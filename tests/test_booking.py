from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.repositories.postgres import SessionLocal
from app.models.booking import Booking
from app.services.memory_service import clear_history


client = TestClient(app)


def _unique_identity() -> tuple[str, str]:
    """
    Generate a unique (name, email) pair per test run.

    Why: hardcoded values cause rows to collide across runs and make
    cleanup unreliable. Unique values give every test its own data.
    """
    suffix = uuid4().hex[:8]
    return f"Test User {suffix}", f"user-{suffix}@example.com"


def _mock_llm(monkeypatch, responses: dict):
    """
    Replace the LLM field-extraction with a scripted response.

    Why mock: unit tests must be fast, deterministic, and independent
    of Google's quota. The LLM is exercised in the live app and in the
    marked integration test, not here.
    """
    def fake_extract(message: str) -> dict:
        return responses.get(message, {})

    monkeypatch.setattr(
        "app.services.booking_service._extract_fields_with_llm",
        fake_extract,
    )


def test_booking_complete_flow(monkeypatch):
    """Happy path: four messages fill the four required fields."""
    session_id = f"booking-{uuid4()}"
    name, email = _unique_identity()
    clear_history(session_id)

    _mock_llm(monkeypatch, {
        "I want to book an interview": {},
        f"My name is {name}": {"name": name},
        f"Email is {email}": {"email": email},
        "Date 2026-12-01": {"date": "2026-12-01"},
        "Time 14:00": {"time": "14:00"},
    })

    r1 = client.post("/chat", json={"session_id": session_id, "message": "I want to book an interview"})
    assert r1.status_code == 200
    assert "name" in r1.json()["answer"].lower()

    client.post("/chat", json={"session_id": session_id, "message": f"My name is {name}"})
    client.post("/chat", json={"session_id": session_id, "message": f"Email is {email}"})
    client.post("/chat", json={"session_id": session_id, "message": "Date 2026-12-01"})
    r5 = client.post("/chat", json={"session_id": session_id, "message": "Time 14:00"})

    assert r5.status_code == 200
    assert "booked" in r5.json()["answer"].lower()

    # Verify persistence, then clean up only this run's row.
    db = SessionLocal()
    try:
        rows = db.query(Booking).filter(Booking.email == email).all()
        assert len(rows) == 1
        assert rows[0].name == name
        db.query(Booking).filter(Booking.email == email).delete()
        db.commit()
    finally:
        db.close()

    clear_history(session_id)


def test_booking_rejects_invalid_email(monkeypatch):
    """An invalid email should be rejected after all fields are collected."""
    session_id = f"booking-{uuid4()}"
    name, _ = _unique_identity()
    clear_history(session_id)

    _mock_llm(monkeypatch, {
        "book an interview": {},
        f"name is {name}": {"name": name},
        "email is not-an-email": {"email": "not-an-email"},
        "2026-12-01": {"date": "2026-12-01"},
        "14:00": {"time": "14:00"},
    })

    client.post("/chat", json={"session_id": session_id, "message": "book an interview"})
    client.post("/chat", json={"session_id": session_id, "message": f"name is {name}"})
    client.post("/chat", json={"session_id": session_id, "message": "email is not-an-email"})
    client.post("/chat", json={"session_id": session_id, "message": "2026-12-01"})
    r = client.post("/chat", json={"session_id": session_id, "message": "14:00"})

    body = r.json()["answer"].lower()
    assert "email" in body
    assert "valid" in body or "@" in body

    db = SessionLocal()
    try:
        rows = db.query(Booking).filter(Booking.name == name).all()
        assert len(rows) == 0
    finally:
        db.close()

    clear_history(session_id)


def test_booking_rejects_past_date(monkeypatch):
    """A past date should be rejected once all four fields are present."""
    session_id = f"booking-{uuid4()}"
    name, email = _unique_identity()
    clear_history(session_id)

    _mock_llm(monkeypatch, {
        "I want to book an interview": {},
        f"My name is {name}": {"name": name},
        email: {"email": email},
        "2020-01-01": {"date": "2020-01-01"},
        "14:00": {"time": "14:00"},
    })

    client.post("/chat", json={"session_id": session_id, "message": "I want to book an interview"})
    client.post("/chat", json={"session_id": session_id, "message": f"My name is {name}"})
    client.post("/chat", json={"session_id": session_id, "message": email})
    client.post("/chat", json={"session_id": session_id, "message": "2020-01-01"})
    # Send time too — all four fields now present, so validation runs.
    r = client.post("/chat", json={"session_id": session_id, "message": "14:00"})

    assert r.status_code == 200
    assert "past" in r.json()["answer"].lower()

    # Confirm nothing was written.
    db = SessionLocal()
    try:
        rows = db.query(Booking).filter(Booking.email == email).all()
        assert len(rows) == 0
    finally:
        db.close()

    clear_history(session_id)


def test_booking_rejects_non_business_time(monkeypatch):
    """A time outside 09:00–17:00 should be rejected."""
    session_id = f"booking-{uuid4()}"
    name, email = _unique_identity()
    clear_history(session_id)

    _mock_llm(monkeypatch, {
        "I want to book an interview": {},
        f"My name is {name}": {"name": name},
        email: {"email": email},
        "2026-12-01": {"date": "2026-12-01"},
        "03:00": {"time": "03:00"},
    })

    client.post("/chat", json={"session_id": session_id, "message": "I want to book an interview"})
    client.post("/chat", json={"session_id": session_id, "message": f"My name is {name}"})
    client.post("/chat", json={"session_id": session_id, "message": email})
    client.post("/chat", json={"session_id": session_id, "message": "2026-12-01"})
    r = client.post("/chat", json={"session_id": session_id, "message": "03:00"})

    assert r.status_code == 200
    body = r.json()["answer"].lower()
    assert "09:00" in body or "between" in body

    db = SessionLocal()
    try:
        rows = db.query(Booking).filter(Booking.email == email).all()
        assert len(rows) == 0
    finally:
        db.close()

    clear_history(session_id)


def test_booking_rejects_short_name(monkeypatch):
    """A name without letters (e.g. '12') should be rejected."""
    session_id = f"booking-{uuid4()}"
    _, email = _unique_identity()
    clear_history(session_id)

    _mock_llm(monkeypatch, {
        "I want to book an interview": {},
        "name is 12": {"name": "12"},
        email: {"email": email},
        "2026-12-01": {"date": "2026-12-01"},
        "14:00": {"time": "14:00"},
    })

    client.post("/chat", json={"session_id": session_id, "message": "I want to book an interview"})
    client.post("/chat", json={"session_id": session_id, "message": "name is 12"})
    client.post("/chat", json={"session_id": session_id, "message": email})
    client.post("/chat", json={"session_id": session_id, "message": "2026-12-01"})
    r = client.post("/chat", json={"session_id": session_id, "message": "14:00"})

    assert r.status_code == 200
    body = r.json()["answer"].lower()
    assert "name" in body
    assert "letter" in body or "empty" in body

    clear_history(session_id)