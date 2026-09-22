from fastapi import APIRouter, HTTPException

from app.models.chat import ChatRequest, ChatResponse, SourceReference
from app.services.booking_service import (
    get_booking_state,
    handle_booking_message,
    is_booking_intent,
)
from app.services.memory_service import append_turn, get_history
from app.services.rag_service import answer_question


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational endpoint with two modes:

    - Booking mode: entered when a booking is already in progress or
      the message matches a booking keyword. Handled by booking_service.
    - RAG mode: everything else. Retrieval + LLM with conversation memory.
    """
    try:
        # Load prior turns so the LLM can resolve follow-up references.
        history = get_history(request.session_id)

        # Route to booking flow when either a booking is in progress
        # or this message starts one. Once routing happens, RAG is
        # skipped entirely for this turn.
        in_booking = get_booking_state(request.session_id) is not None
        booking_intent = is_booking_intent(request.message)

        if in_booking or booking_intent:
            answer = handle_booking_message(request.session_id, request.message)
            sources: list[SourceReference] = []
        else:
            result = answer_question(
                question=request.message,
                limit=3,
                history=history,
            )
            answer = result["answer"]
            sources = [
                SourceReference(
                    filename=s.get("filename", "unknown"),
                    chunk_index=s.get("chunk_index", 0),
                    score=s.get("score", 0.0),
                    text=s.get("text", ""),
                )
                for s in result["sources"]
            ]

        # Save the user message first, then the assistant reply.
        # This runs for both modes so chat history stays complete.
        append_turn(request.session_id, "user", request.message)
        append_turn(request.session_id, "assistant", answer)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        sources=sources,
    )