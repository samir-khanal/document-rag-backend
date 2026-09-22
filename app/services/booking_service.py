import json
import re
from datetime import date, datetime, time
from uuid import uuid4

from pydantic import BaseModel, EmailStr, ValidationError, field_validator

from app.repositories.postgres import SessionLocal, save_booking
from app.repositories.redis import redis_client


# How long an in-progress booking survives in Redis.
# Same idea as chat history: bookings that aren't finished within
# an hour are almost certainly abandoned and should be cleaned up.
BOOKING_STATE_TTL_SECONDS = 60 * 60


# Keywords that put a session into booking mode. Kept simple on purpose:
# the company asked for LLM extraction of *fields*, not intent detection.
# A keyword list is cheap, deterministic, and easy to test.
BOOKING_KEYWORDS = [
    "book an interview",
    "book interview",
    "schedule an interview",
    "schedule interview",
    "book a meeting",
    "schedule a meeting",
]


# Field order = the order we ask for them.
BOOKING_FIELDS = ["name", "email", "date", "time"]


class BookingData(BaseModel):
    """
    Validated booking payload.

    Pydantic does the hard work: if the LLM returns a malformed email,
    a past date, or a non-business time, validation fails and the user
    is asked again. This is the "don't blindly trust the LLM" rule
    from the task.
    """

    name: str
    email: EmailStr
    date: str
    time: str

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty.")
        # Reject inputs like "123" or "!!!" — a name should have letters.
        if not any(ch.isalpha() for ch in v):
            raise ValueError("Name must contain letters.")
        if len(v) < 2:
            raise ValueError("Name is too short.")
        return v

    @field_validator("date")
    @classmethod
    def date_must_parse_and_be_future(cls, v: str) -> str:
        # Accept ISO format (2026-10-05) or common US format (10/05/2026).
        parsed = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(v, fmt).date()
                break
            except ValueError:
                continue

        if parsed is None:
            raise ValueError("Date must be YYYY-MM-DD or MM/DD/YYYY.")

        # You cannot book an interview for a day that already happened.
        if parsed < date.today():
            raise ValueError("Interview date cannot be in the past.")

        # Reject far-future dates — a typo like 2099 is not a real booking.
        days_ahead = (parsed - date.today()).days
        if days_ahead > 365:
            raise ValueError("Interview date must be within the next year.")

        return v

    @field_validator("time")
    @classmethod
    def time_must_parse_and_be_business_hours(cls, v: str) -> str:
        # Accept 24h ("14:00") or 12h with am/pm ("2:00pm").
        cleaned = v.strip().upper().replace(" ", "")
        parsed = None
        for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
            try:
                parsed = datetime.strptime(cleaned, fmt).time()
                break
            except ValueError:
                continue

        if parsed is None:
            raise ValueError("Time must be HH:MM (24h) or h:MMam/pm.")

        # Keep interview times realistic — 9 AM to 5 PM.
        if not (9 <= parsed.hour < 17):
            raise ValueError("Interview time must be between 09:00 and 17:00.")

        return v


def _state_key(session_id: str) -> str:
    """Redis key for the in-progress booking state."""
    return f"booking:state:{session_id}"


def get_booking_state(session_id: str) -> dict | None:
    """
    Return the in-progress booking state, or None if this session
    isn't currently booking anything.
    """
    raw = redis_client.get(_state_key(session_id))
    if not raw:
        return None
    return json.loads(raw)


def _set_booking_state(session_id: str, state: dict) -> None:
    """Save the in-progress booking state with a rolling TTL."""
    redis_client.set(
        _state_key(session_id),
        json.dumps(state),
        ex=BOOKING_STATE_TTL_SECONDS,
    )


def _clear_booking_state(session_id: str) -> None:
    """Delete the booking state once it's finished or cancelled."""
    redis_client.delete(_state_key(session_id))


def _next_missing_field(state: dict) -> str | None:
    """Return the first field still missing from the booking data."""
    data = state.get("data", {})
    for field in BOOKING_FIELDS:
        if not data.get(field):
            return field
    return None


def is_booking_intent(message: str) -> bool:
    """
    Cheap keyword check for booking intent.

    Why not use the LLM here: intent classification on every message
    would burn quota and add latency to normal questions. A keyword
    list is deterministic, testable, and good enough for this project.
    """
    lower = message.lower()
    return any(keyword in lower for keyword in BOOKING_KEYWORDS)


def _extract_fields_with_llm(message: str) -> dict:
    """
    Extract booking fields from a message.

    Primary path: call Gemini to parse free-text and return structured
    fields. Fallback path: if Gemini is unavailable (quota exhausted,
    network error, etc.), use regex so booking still works.

    The backend validation layer is identical in both cases — the
    fallback only affects how fields are extracted, not whether they
    are trusted.
    """
    from google.genai import errors
    from app.services.llm_service import client, MODEL_NAME

    prompt = (
        "Extract interview booking details from the message below.\n"
        "\n"
        'Return ONLY a JSON object with any of these keys present: '
        '"name", "email", "date", "time".\n'
        "- date must be YYYY-MM-DD\n"
        "- time must be HH:MM 24-hour\n"
        "- omit keys that are not present\n"
        "- output ONLY the JSON object\n"
        "\n"
        f"Message:\n{message}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        text = (response.text or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}

    except (errors.ClientError, errors.ServerError):
        # Gemini unavailable (usually 429 quota) — fall back to regex.
        # This keeps the demo working even when the free tier is exhausted.
        return _extract_fields_with_regex(message)


def _extract_fields_with_regex(message: str) -> dict:
    """
    Regex fallback for when the LLM is unavailable.

    Handles two kinds of input:
    - Labelled: "My name is Samir", "email is x@y.com"
    - Bare values: "Samir Khanal", "x@y.com", "2026-10-05", "14:00"

    The bare-value case matters because the assistant asks one field
    at a time — users reply with just the value, not a full sentence.
    """
    result: dict = {}
    text = message.strip()

    # Email — any email-shaped token
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        result["email"] = email_match.group(0)

    # Date — ISO format
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        result["date"] = date_match.group(0)

    # Time — HH:MM, optionally followed by am/pm
    time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", text, re.IGNORECASE)
    if time_match:
        result["time"] = time_match.group(1).strip()

    # Name — labelled form ("my name is ...", "i am ...")
    name_match = re.search(
        r"(?:my name is|i am|i'm|name[:\s]+)\s*([A-Za-z][A-Za-z\s\.\-']{1,60})",
        text,
        re.IGNORECASE,
    )
    if name_match:
        result["name"] = name_match.group(1).strip()
    elif not result and re.fullmatch(r"[A-Za-z][A-Za-z\s\.\-']{1,60}", text):
        # Bare name — no other field was found and the whole message
        # looks like a name. Handles the common case where the user
        # replies to "What is your full name?" with just "Samir Khanal".
        result["name"] = text

    return result


def _prompt_for(field: str) -> str:
    """Human-readable question for a missing field."""
    prompts = {
        "name": "What is your full name?",
        "email": "What is your email address?",
        "date": "What date would you like? (YYYY-MM-DD)",
        "time": "What time would you like? (HH:MM, 24-hour)",
    }
    return prompts[field]


def handle_booking_message(session_id: str, message: str) -> str:
    """
    Drive the booking conversation state machine.

    Called from /chat when either:
      - the message looks like a booking request, or
      - a booking is already in progress for this session.

    Returns the assistant's next message — either a prompt for the
    next missing field, a reason + re-prompt when a field fails
    validation, or a confirmation once everything is saved.
    """
    state = get_booking_state(session_id)

    # Not in progress yet → start a new booking.
    if state is None:
        state = {"data": {}}

    # Ask the LLM to pull any fields out of this message.
    extracted = _extract_fields_with_llm(message)

    # Merge into the state. Existing values are preserved unless the
    # new extraction produced a value for that field.
    data = state.get("data", {})
    for field in BOOKING_FIELDS:
        if field in extracted and extracted[field]:
            data[field] = extracted[field]
    state["data"] = data

    # Still missing something? Ask for it and wait for the next message.
    missing = _next_missing_field(state)
    if missing:
        _set_booking_state(session_id, state)
        return _prompt_for(missing)

    # All four fields present — validate with Pydantic.
    try:
        validated = BookingData(**data)
    except ValidationError as exc:
        # Keep the fields that were already valid; clear only the ones
        # that failed. The user re-enters only what's wrong instead of
        # restarting the entire booking.
        bad_fields = {
            err["loc"][0]
            for err in exc.errors()
            if err.get("loc")
        }
        for field in bad_fields:
            data.pop(field, None)

        state["data"] = data
        _set_booking_state(session_id, state)

        # Re-ask for the first field that failed, prepending the reason
        # so the user knows why they are being asked again.
        next_field = _next_missing_field(state) or next(iter(bad_fields))

        # Pydantic's loc entries can be ints or strings. Normalize to str
        # so the lookup below has a consistent type to compare against.
        next_field_str = str(next_field)

        reason = next(
            (
                err["msg"]
                for err in exc.errors()
                if err.get("loc") and str(err["loc"][0]) == next_field_str
            ),
            "That value didn't look right.",
        )

        return f"{reason} {_prompt_for(next_field_str)}"

    # Persist to PostgreSQL. Redis is cleared because the booking is done.
    booking_id = uuid4()

    # Convert validated strings into Python date/time objects for the DB.
    parsed_date = _parse_date(validated.date)
    parsed_time = _parse_time(validated.time)

    db = SessionLocal()
    try:
        save_booking(
            db=db,
            booking_id=booking_id,
            name=validated.name,
            email=validated.email,
            interview_date=parsed_date,
            interview_time=parsed_time,
        )
    finally:
        db.close()

    _clear_booking_state(session_id)

    return (
        f"Your interview is booked for {validated.date} at {validated.time}. "
        f"A confirmation will be sent to {validated.email}. "
        f"Your booking ID is {booking_id}."
    )


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unparseable date: {value}")


def _parse_time(value: str) -> time:
    cleaned = value.strip().upper().replace(" ", "")
    for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Unparseable time: {value}")