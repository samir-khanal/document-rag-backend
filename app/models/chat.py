from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        description="Groups messages into one conversation.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's message.",
    )


class SourceReference(BaseModel):
    """One retrieved chunk, shown so answers can be traced back."""
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceReference]