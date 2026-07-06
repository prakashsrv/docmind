import uuid

from app.core import config
from app.models.chunk import Chunk
from app.models.document import Document


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window split. Pure string logic, no knowledge of Document/Chunk."""
    step = chunk_size - overlap
    parts = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        parts.append(text[start:end])
        start += step

    return parts


def chunk_document(document: Document, chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    """Split a Document's text into overlapping Chunk objects.

    With chunk_size=500 and overlap=100, each chunk is 500 characters and
    starts 400 characters after the previous one -- so the last 100
    characters of one chunk are repeated at the start of the next. The
    overlap exists so a sentence or idea that straddles a chunk boundary
    still appears in full in at least one chunk, instead of being cut in
    half and losing context for retrieval later.

    Each Chunk keeps document_id (which Document it came from) and
    chunk_index (its position). Phase 3 needs both to store embeddings
    and trace a search result back to its source document and location.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    parts = _split_text(document.text, chunk_size, overlap)

    return [
        Chunk(
            id=str(uuid.uuid4()),
            document_id=document.id,
            chunk_index=index,
            text=part,
        )
        for index, part in enumerate(parts)
    ]
