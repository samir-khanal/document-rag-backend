from google import genai

from app.core.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)

MODEL_NAME = "gemini-3.6-flash"


def _format_history(history: list[dict]) -> str:
    """
    Turn the stored chat turns into a plain-text block.

    Each item is expected to look like {"role": "user"|"assistant",
    "content": "..."}. Anything else is skipped so a malformed Redis
    entry can never crash the LLM call.
    """
    lines: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    """Generate an answer using Gemini, retrieved context, and prior turns."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    # Prior turns are optional. On the first message of a session the
    # history is empty, so the prompt behaves exactly as in Phase 3.
    # Once history exists, follow-up questions like "What about damaged
    # items?" are interpreted in the context of the earlier exchange.
    history_block = _format_history(history or [])

    if history_block:
        history_section = f"""
Previous conversation (for context only — do not repeat it):
{history_block}
"""
    else:
        history_section = ""

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using only the information provided
in the context below.

If the answer cannot be found in the context, say that you
could not find the answer in the provided documents.

Do not invent or assume information that is not present in
the context.

Use the previous conversation only to understand what the user
is referring to — for example, a follow-up question may use
"it" or "that" to mean something discussed earlier. The actual
answer must still come from the context.
{history_section}
Context:
{context}

User question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise ValueError("The language model returned an empty response.")

    return response.text.strip()