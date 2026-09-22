import json

from app.repositories.redis import redis_client


# Cap and TTL keep Redis bounded. Without them, an abandoned session
# would grow forever and every new turn would ship the entire history
# to the LLM.
MAX_HISTORY_TURNS = 10
SESSION_TTL_SECONDS = 60 * 60 * 24   # 24 hours


def _history_key(session_id: str) -> str:
    """Single source of truth for the Redis key format."""
    return f"chat:history:{session_id}"


def get_history(session_id: str) -> list[dict]:
    """
    Load a session's chat history, oldest first.

    Returns an empty list for a new or expired session — this is the
    normal case on the first message, not an error.
    """
    raw_items = redis_client.lrange(_history_key(session_id), 0, -1)
    return [json.loads(item) for item in raw_items]


def append_turn(session_id: str, role: str, content: str) -> None:
    """Add one user or assistant message to the session."""
    if role not in {"user", "assistant"}:
        raise ValueError(f"Invalid role: {role}")

    key = _history_key(session_id)
    redis_client.rpush(key, json.dumps({"role": role, "content": content}))

    # LTRIM keeps only the most recent N messages. TTL is refreshed
    # on every write so an active chat never expires mid-conversation,
    # but an abandoned one is cleaned up automatically.
    redis_client.ltrim(key, -MAX_HISTORY_TURNS * 2, -1)
    redis_client.expire(key, SESSION_TTL_SECONDS)


def clear_history(session_id: str) -> None:
    """Delete all turns for a session (used in tests and /reset)."""
    redis_client.delete(_history_key(session_id))