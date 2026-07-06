from app.core import config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into fixed-size, overlapping chunks.

    With chunk_size=500 and overlap=100, each chunk is 500 characters and
    starts 400 characters after the previous one -- so the last 100
    characters of one chunk are repeated at the start of the next. The
    overlap exists so a sentence or idea that straddles a chunk boundary
    still appears in full in at least one chunk, instead of being cut in
    half and losing context for retrieval later.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step

    return chunks
