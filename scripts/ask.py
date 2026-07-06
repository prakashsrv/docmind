"""Phase 4 Step 2: ask a question against the persistent Chroma store --
no PDF parsing or re-embedding at query time. This is the "Load Chroma ->
Ready" startup path the mentor plan describes, as opposed to
scripts/test_rag.py, which rebuilds an in-memory store from scratch every
run.

Run scripts/ingest.py at least once before this for any document you want
searchable.

Usage:
    python scripts/ask.py "your question here"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.rag_service import RagService
from app.retrieval.retriever import Retriever
from app.storage.chroma_store import ChromaVectorStore


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]

    store = ChromaVectorStore()
    print(f"Loaded Chroma store ({store.collection.count()} chunks) -- ready.\n")

    rag_service = RagService(Retriever(store))
    answer = rag_service.answer(question)

    print(f"Question: {question}\n")
    print(answer)


if __name__ == "__main__":
    main()
