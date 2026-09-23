# Document RAG Backend

A modular FastAPI backend implementing document ingestion, semantic retrieval, conversational RAG, Redis-based multi-turn memory, and natural-language interview booking.

Built as an AI/ML backend take-home project with:

- PDF and TXT document ingestion
- Fixed-size and recursive chunking
- Sentence Transformer embeddings
- Qdrant semantic retrieval
- Custom RAG without `RetrievalQAChain`
- Redis-based multi-turn conversation memory
- LLM-assisted interview booking
- PostgreSQL persistence
- Pydantic validation
- Gemini fallback for booking extraction failures
- Automated unit and integration testing
- Docker Compose development environment
- Swagger/OpenAPI documentation

## Overview

The backend exposes two REST APIs:

1. **Document Ingestion** — accepts PDF/TXT files, extracts and chunks text, generates embeddings, and stores vectors and metadata.

2. **Conversational RAG** — retrieves relevant document chunks, uses conversation history to handle follow-up questions, and routes interview-booking requests through a separate booking workflow.

Both are implemented and runnable with a single `docker compose up -d`.

---

## Architecture

```
                       Client
                         │
                         │ HTTP
                         ▼
                    ┌─────────┐
                    │ FastAPI │
                    └────┬────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   /documents/ingest                  /chat
          │                             │
   extract PDF/TXT              load history from Redis
          │                             │
       chunk                       intent check
    ┌─────┴─────┐                ┌──────┴──────┐
    │           │                │             │
  Fixed     Recursive          RAG          Booking
    │           │                │             │
    └─────┬─────┘          retrieve chunks   LLM extracts
          │                       │          fields
     embeddings                   │             │
          │                    Qdrant           │
          ▼                       │          validate
       Qdrant ◄───────────────  LLM           │
          │                       │       PostgreSQL
          │                    answer           │
          └── metadata ───► PostgreSQL          │
                                     │          │
                                     ▼          ▼
                                  Redis     confirmation
                                (history)   (booking_id)
```

The two paths share infrastructure but never mix: RAG looks up documents, booking collects and stores a record. An intent check at `/chat` decides which path runs.

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Web framework | FastAPI | Async, Pydantic-native, auto-generated docs at `/docs` |
| Vector store | Qdrant | Runs locally in Docker — no paid account needed |
| Metadata DB | PostgreSQL | Structured data that must survive restarts |
| Chat memory | Redis | Ephemeral; TTL cleans up abandoned sessions |
| Embeddings | `all-MiniLM-L6-v2` | Small (90 MB), CPU-only, 384-dim vectors |
| LLM | Gemini (`gemini-3.6-flash`) | Free tier, reliable structured output |
| PDF extraction | PyMuPDF | Fast, no system dependencies |
| Chunking | Fixed + Recursive | Two selectable strategies as required |
| Validation | Pydantic | Email, date, and range checks |
| Containers | Docker Compose | One command starts Postgres, Redis, Qdrant |

---

## Key Design Decisions

### Qdrant for vector storage

Qdrant was selected as the vector database because the task required a vector store and explicitly excluded FAISS and Chroma. It supports similarity search with metadata payloads and can run locally through Docker.

### Redis for conversation state

Redis stores short-lived conversation history and in-progress booking state. TTLs prevent abandoned sessions from remaining indefinitely.

### PostgreSQL for persistent data

PostgreSQL stores document metadata and completed bookings because these records require structured persistence beyond the lifetime of a chat session.

### Custom RAG pipeline

The retrieval and generation steps are implemented directly rather than using `RetrievalQAChain`:

`query → embedding → Qdrant search → context construction → prompt → LLM → response`

This keeps retrieval, prompt construction, memory, and response handling under application control.

### Separate conversation history from retrieval

Conversation history is provided to the LLM so it can interpret follow-up questions, but retrieval is based on the current user message. This prevents unrelated previous turns from continually accumulating into the retrieval query.

## Setup

### Requirements

- Docker Desktop
- Python 3.13
- A Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Steps

1. Clone and enter the project:

```bash
git clone <your-repo-url>
cd palm-mind-ai-task
```

2. Create the virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql+psycopg://app_user:app_password@localhost:5432/app_db
DATABASE_URL=postgresql+psycopg://app_user:app_password@localhost:5432/app_db
REDIS_URL=redis://localhost:6379/0
```

Replace `changeme` with the Postgres password from `docker-compose.yml`. `POSTGRES_URL` and `DATABASE_URL` are identical on purpose — the app reads `POSTGRES_URL` via `config.py`, while some SQLAlchemy tooling expects `DATABASE_URL`.

4. Start the services:

```bash
docker compose up -d
```

5. Wait ~10 seconds for Postgres to finish initializing, then create the tables:

```bash
python -m app.db.init_db
```

6. Run the API:

```bash
uvicorn app.main:app --reload
```

Swagger UI is at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API overview

### `POST /documents/ingest`

Upload a `.pdf` or `.txt` file and pick a chunking strategy (`fixed` or `recursive`).

Supported formats are PDF and TXT only. Any filename works — content is extracted, chunked, and embedded. You can ingest multiple files; retrieval spans all of them.

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@data/retail_policy.txt" \
  -F "chunking_strategy=recursive"
```

### `POST /chat`

Handles both RAG questions and interview bookings. If the message matches a booking keyword, or if a booking is already in progress for that `session_id`, the booking flow runs. Otherwise it's treated as a RAG question.

**RAG example:**

```json
{
  "session_id": "demo-1",
  "message": "How long do I have to return a product?"
}
```

**Booking example** — five messages with the same `session_id`:

```json
{ "session_id": "book-1", "message": "I want to book an interview" }
{ "session_id": "book-1", "message": "My name is Samir Khanal" }
{ "session_id": "book-1", "message": "samir@example.com" }
{ "session_id": "book-1", "message": "2026-12-01" }
{ "session_id": "book-1", "message": "14:00" }
```

The 5th call returns a confirmation with a booking UUID. If a field fails validation (bad email, past date, non-business time), the response names the reason and re-asks only that field — the other three stay filled.

---

## RAG flow

1. Load prior turns from Redis for the `session_id`.
2. Embed the current message with the same model used during ingestion.
3. Qdrant returns the top-K similar chunks, deduplicated by `(filename, chunk_index)`.
4. Chunks are formatted into a context block with `[Source N: filename]` markers.
5. Gemini receives the context, the current question, and the conversation history.
6. The answer is returned with the retrieved sources.
7. Both turns are appended to Redis.

History is used to interpret follow-up questions like *"what about damaged items?"*. Retrieval itself uses only the current message, so the same question always returns the same chunks.

---

## Chunking strategies

**Fixed** splits text into windows of a fixed character size with overlap. Predictable and good for uniformly structured documents. The overlap is snapped to the nearest word boundary so chunks don't start mid-word.

**Recursive** tries progressively smaller separators — paragraphs, then sentences, then words. This preserves natural text boundaries and is useful for prose-heavy documents such as policies and FAQs. This is the default strategy.

Both strategies use the same embedding model and store the same metadata, so the retrieval path is identical regardless of which one produced the chunks.

---

## Booking flow

1. Keyword check enters booking mode (or continues it if already in progress).
2. The LLM extracts any of the four fields present in the message — `name`, `email`, `date`, `time`.
3. Extracted fields merge into the booking state in Redis.
4. If a field is missing, the assistant asks for it.
5. Once all four are present, Pydantic validates them. Invalid fields are cleared and the user is asked again with the specific reason.
6. On success, the booking is written to Postgres and Redis state is deleted.

Validation rules:

- **Name** — not empty, contains letters, length ≥ 2
- **Email** — valid format via Pydantic's `EmailStr`
- **Date** — parseable, not in the past, within one year
- **Time** — parseable, between 09:00 and 17:00

If Gemini is unavailable (429 or 503), a regex fallback extracts the same fields so the flow keeps working. Validation is identical either way.

### Where bookings are stored

Bookings go into the `bookings` table in **PostgreSQL**, not Redis.

Redis holds only the **in-progress** booking state while fields are being collected. Once validation passes, `save_booking()` inserts the row into Postgres and clears Redis.

This split matters: if Redis restarts mid-conversation, the user loses their progress but no completed booking is lost. If Postgres restarts, completed bookings survive.

Verify a saved booking:

```bash
docker exec -it palm-mind-postgres psql -U samir -d palm_mind \
  -c "SELECT booking_id, name, email, interview_date, interview_time FROM bookings;"
```

---

## Testing

```bash
pytest tests/ -v
```

Expected output:

```
22 passed, 1 deselected
```

The deselected test (`tests/test_rag_manual.py::test_rag_answer`) is marked as integration because it calls the real Gemini API. It's excluded by default so quota errors or Google outages never break the everyday test run.

Run it explicitly with:

```bash
pytest tests/ -v -m integration
```

What's covered:

| Area | Tests |
|------|-------|
| Chunking | Fixed and recursive strategies, empty input |
| Embeddings | Text → vector, batch, determinism |
| Qdrant | Collection creation, chunk storage, retrieval |
| Postgres | Document metadata persistence |
| Redis memory | Turn storage, multi-turn history |
| Chat | Single-turn, multi-turn, history passed to LLM |
| Booking | Full flow, invalid email, past date, non-business time, short name |
| Ingest | Unsupported extension, empty file, whitespace-only file |

Chat and booking tests mock the LLM for speed and determinism. Real LLM behavior is covered by the marked integration test.

---

## Project layout

```
app/
├── api/              # HTTP endpoints
│   ├── documents.py
│   └── chat.py
├── core/
│   └── config.py     # Settings loaded from .env
├── db/
│   ├── base.py
│   └── init_db.py
├── models/
│   ├── document.py
│   ├── booking.py
│   └── chat.py
├── repositories/     # External I/O
│   ├── postgres.py
│   ├── qdrant.py
│   └── redis.py
└── services/         # Business logic
    ├── chunking.py
    ├── document_service.py
    ├── embedding_service.py
    ├── retrieval_service.py
    ├── rag_service.py
    ├── llm_service.py
    ├── memory_service.py
    └── booking_service.py

data/
└── retail_policy.txt  # Sample document for the demo
```

The layering rule: `api/` handles HTTP and request validation, `services/` handles logic, `repositories/` talks to external systems. A service never imports FastAPI; an endpoint never imports the Redis client directly.

---

## Production Considerations

- **No authentication.** `session_id` is client-supplied. In production it would be tied to an authenticated user.
- **No rate limiting.** A single client can exhaust the Gemini quota.
- **Character-based chunking.** Token-aware chunking would handle code and non-English text better.
- **Keyword-based intent detection.** A message like "I'd like to schedule something" without the word "interview" won't enter booking mode. An LLM classifier would fix this but costs a request per message.
- **No retry on Postgres write failure.** A brief database outage during a booking save loses that booking. A retry with a dead-letter queue would be the production answer.

---

## Quick verification

1. `docker compose up -d`
2. `python -m app.db.init_db`
3. `uvicorn app.main:app --reload`
4. Open `http://localhost:8000/docs`
5. Upload `data/retail_policy.txt` via `/documents/ingest`
6. Ask `"How long do I have to return a product?"` via `/chat`
7. Run the five-message booking flow on the same endpoint

That covers both APIs and shows the routing between RAG and booking.
