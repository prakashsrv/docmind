"""End-to-end test for Phase 4: ingest a PDF into an in-memory vector
store, then ask RagService a question and print the grounded answer with
sources.

Usage:
    python scripts/test_rag.py data/documents/your_file.pdf "your question"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding.vector_store import InMemoryVectorStore
from app.ingestion.pipeline import process_pdf
from app.retrieval.rag_service import RagService
from app.retrieval.retriever import Retriever


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/test_rag.py <path-to-pdf> "<question>"')
        sys.exit(1)

    path = sys.argv[1]
    question = sys.argv[2]

    _, chunks = process_pdf(path)

    store = InMemoryVectorStore()
    store.add(chunks)

    rag_service = RagService(Retriever(store))
    answer = rag_service.answer(question)

    print(f"Question: {question}\n")
    print(answer)


if __name__ == "__main__":
    main()
