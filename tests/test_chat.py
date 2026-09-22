from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.services.memory_service import clear_history, get_history


client = TestClient(app)


def test_chat_stores_turns_in_redis(monkeypatch):
    """One message should produce exactly two stored turns."""
    session_id = f"test-{uuid4()}"

    # Patch the LLM call so this test never depends on Google's servers.
    # The test's job is to verify Redis memory, not generation.
    def fake_answer_question(question, limit=3, history=None):
        return {"answer": "fake answer", "sources": []}

    monkeypatch.setattr("app.api.chat.answer_question", fake_answer_question)

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "How long do I have to return a product?",
            },
        )
        assert response.status_code == 200

        history = get_history(session_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    finally:
        clear_history(session_id)


def test_chat_multiple_turns_share_history(monkeypatch):
    """Two messages with the same session_id should produce four turns."""
    session_id = f"test-{uuid4()}"

    def fake_answer_question(question, limit=3, history=None):
        return {"answer": f"fake answer to: {question}", "sources": []}

    monkeypatch.setattr("app.api.chat.answer_question", fake_answer_question)

    try:
        client.post("/chat", json={
            "session_id": session_id,
            "message": "How long do I have to return a product?",
        })
        client.post("/chat", json={
            "session_id": session_id,
            "message": "What about damaged items?",
        })

        history = get_history(session_id)
        assert len(history) == 4
        assert [h["role"] for h in history] == [
            "user", "assistant", "user", "assistant",
        ]

    finally:
        clear_history(session_id)


def test_chat_passes_history_to_llm(monkeypatch):
    """
    On a second message within the same session, the LLM should
    receive the previous turns — this is what makes follow-up
    questions work.

    We capture what answer_question receives instead of calling the
    real LLM, so the assertion is deterministic.
    """
    session_id = f"test-{uuid4()}"
    captured: dict = {}

    def fake_answer_question(question, limit=3, history=None):
        captured["question"] = question
        captured["history"] = history
        return {"answer": "fake answer", "sources": []}

    monkeypatch.setattr("app.api.chat.answer_question", fake_answer_question)

    try:
        # First turn — history should be empty.
        client.post("/chat", json={
            "session_id": session_id,
            "message": "How long do I have to return a product?",
        })
        assert captured["history"] == []

        # Second turn — history should contain the first user message
        # plus the fake assistant reply.
        client.post("/chat", json={
            "session_id": session_id,
            "message": "What about damaged items?",
        })
        assert len(captured["history"]) == 2
        assert captured["history"][0]["role"] == "user"
        assert captured["history"][0]["content"] == "How long do I have to return a product?"
        assert captured["history"][1]["role"] == "assistant"

    finally:
        clear_history(session_id)