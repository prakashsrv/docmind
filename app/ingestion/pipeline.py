import os
import uuid

from app.embedding.embedding_service import EmbeddingService
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import load_pdf_text
from app.models.chunk import Chunk
from app.models.document import Document


def load_document(path: str) -> Document:
    """PDF file -> Document. Wraps the raw extracted text with an id and name."""
    text = load_pdf_text(path)
    name = os.path.basename(path)
    return Document(id=str(uuid.uuid4()), name=name, text=text)


def process_pdf(path: str) -> tuple[Document, list[Chunk]]:
    """Full ingestion flow: PDF -> Loader -> Document -> Chunker -> EmbeddingService -> Chunks (with embeddings)."""
    document = load_document(path)
    chunks = chunk_document(document)
    chunks = EmbeddingService().embed_chunks(chunks)
    return document, chunks
