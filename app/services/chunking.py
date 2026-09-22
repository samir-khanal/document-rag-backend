from enum import Enum


class ChunkingStrategy(str, Enum):
    """Supported document chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"


def fixed_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split text into fixed-size chunks with overlapping context."""

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[str] = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Keep part of the previous chunk so information near a
        # boundary is not completely lost during retrieval.
        start = end - overlap

    return chunks

def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split text using progressively smaller separators."""

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(value: str, separator_index: int) -> list[str]:
        """Recursively split text until each piece fits the chunk size."""

        if len(value) <= chunk_size:
            return [value.strip()]

        separator = separators[separator_index]

        if separator:
            parts = value.split(separator)
        else:
            # Character-level splitting is the final fallback when
            # no larger meaningful boundary can keep the chunk small enough.
            parts = list(value)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for part in parts:
            part = part.strip()

            if not part:
                continue

            separator_length = len(separator) if current_parts else 0
            proposed_length = current_length + separator_length + len(part)

            if proposed_length <= chunk_size:
                current_parts.append(part)
                current_length = proposed_length
                continue

            if current_parts:
                current_chunk = separator.join(current_parts).strip()
                chunks.append(current_chunk)

            # If this individual piece is still too large, try the next
            # smaller separator instead of cutting it blindly.
            if len(part) > chunk_size and separator_index < len(separators) - 1:
                chunks.extend(split_text(part, separator_index + 1))
                current_parts = []
                current_length = 0
            else:
                current_parts = [part]
                current_length = len(part)

        if current_parts:
            chunks.append(separator.join(current_parts).strip())

        return chunks

    raw_chunks = split_text(text, 0)

    if overlap == 0:
        return raw_chunks

    # Add overlap between neighboring chunks after the recursive split.
    final_chunks: list[str] = []

    for index, chunk in enumerate(raw_chunks):
        if index == 0:
            final_chunks.append(chunk)
            continue

        previous_chunk = raw_chunks[index - 1]
        # Take the last `overlap` characters, then trim to the next word boundary
        raw_overlap = previous_chunk[-overlap:]

        # This avoids cutting a word in half
        # Prefer ending at a sentence; fall back to word boundary
        for delimiter in [". ", "! ", "? ", " "]:
            if delimiter in raw_overlap:
                overlap_text = raw_overlap.split(delimiter, 1)[1]
                break
        else:
            overlap_text = raw_overlap

        final_chunks.append(f"{overlap_text} {chunk}".strip())

    return final_chunks

def chunk_document(
    text: str,
    strategy: ChunkingStrategy,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Chunk a document using the selected strategy."""

    if strategy == ChunkingStrategy.FIXED:
        return fixed_chunking(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if strategy == ChunkingStrategy.RECURSIVE:
        return recursive_chunking(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    raise ValueError(f"Unsupported chunking strategy: {strategy}")