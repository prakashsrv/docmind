"""End-to-end test for Phase 3: ingest a PDF, store its embedded chunks in
the InMemoryVectorStore, then run a semantic search query against it.

Usage:
    python scripts/test_search.py data/documents/resume.pdf "mobile app experience"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding.embedding_service import EmbeddingService
from app.embedding.vector_store import InMemoryVectorStore
from app.ingestion.pipeline import process_pdf


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/test_search.py <path-to-pdf> "<query>"')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]

    _, chunks = process_pdf(path)

    store = InMemoryVectorStore()
    store.add(chunks)

    query_vector = EmbeddingService().embed(query)
    results = store.search_with_scores(query_vector, top_k=5)

    print(f"Query: {query}\n")
    print("Use these scores to calibrate config.SIMILARITY_THRESHOLD -- run a")
    print("few genuinely relevant queries and a few genuinely irrelevant ones")
    print("and see where the scores actually separate.\n")

    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"--- Match {rank} (chunk_index={chunk.chunk_index}, score={score:.3f}) ---")
        print(chunk.text[:300])
        print()


if __name__ == "__main__":
    main()
