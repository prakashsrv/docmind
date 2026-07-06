"""Phase 4 Step 2: ingest a PDF into the persistent Chroma vector store.

Run this once per document (and again whenever a document changes).
Unlike scripts/test_pipeline.py, which builds a throwaway in-memory store
every run, this writes embeddings to disk under config.CHROMA_PERSIST_DIR,
so scripts/ask.py never has to re-parse or re-embed anything at query
time.

Usage:
    python scripts/ingest.py data/documents/your_file.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pipeline import process_pdf
from app.storage.chroma_store import ChromaVectorStore


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <path-to-pdf>")
        sys.exit(1)

    path = sys.argv[1]
    document, chunks = process_pdf(path)

    store = ChromaVectorStore()
    store.add(chunks)

    print(f"Ingested {document.name}")
    print(f"Document id: {document.id}")
    print(f"Chunks embedded and stored: {len(chunks)}")
    print(f"Total chunks now in store: {store.collection.count()}")


if __name__ == "__main__":
    main()
