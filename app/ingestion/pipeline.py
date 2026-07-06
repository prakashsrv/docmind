import hashlib
import os

from app.embedding.embedding_service import EmbeddingService
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import load_pdf_text
from app.models.chunk import Chunk
from app.models.document import Document


def load_document(path: str) -> Document:
    """PDF file -> Document. Wraps the raw extracted text with a stable id
    and name.

    The id is a hash of the extracted text, not a random uuid4, so
    re-ingesting the same PDF (e.g. every time an ingestion script is run)
    produces the same document_id -- and, since chunk ids are derived from
    it too, the same chunk ids. That lets a persistent store like Chroma
    upsert in place instead of accumulating duplicate copies of the same
    document every time ingestion runs.
    """
    text = load_pdf_text(path)
    name = os.path.basename(path)
    document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return Document(id=document_id, name=name, text=text)


def process_pdf(path: str) -> tuple[Document, list[Chunk]]:
    """Full ingestion flow: PDF -> Loader -> Document -> Chunker -> EmbeddingService -> Chunks (with embeddings)."""
    document = load_document(path)
    chunks = chunk_document(document)
    chunks = EmbeddingService().embed_chunks(chunks)
    return document, chunks
