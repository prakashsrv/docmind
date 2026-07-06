"""Phase 4 Step 3/4: compare dense-only retrieval against hybrid
(dense + BM25) retrieval for the same question, so the difference BM25
makes is actually visible -- e.g. a query containing an exact term
(a library name, an acronym) that dense search alone might rank lower.

Usage:
    python scripts/test_hybrid.py data/documents/your_file.pdf "your question"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding.vector_store import InMemoryVectorStore
from app.ingestion.pipeline import process_pdf
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import Retriever


def print_chunks(label, chunks):
    print(f"--- {label} ({len(chunks)} chunks) ---")
    for chunk in chunks:
        print(f"  [Chunk {chunk.chunk_index}] {chunk.text[:80].strip()!r}")
    print()


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/test_hybrid.py <path-to-pdf> "<question>"')
        sys.exit(1)

    path = sys.argv[1]
    question = sys.argv[2]

    _, chunks = process_pdf(path)

    store = InMemoryVectorStore()
    store.add(chunks)

    dense_retriever = Retriever(store)
    bm25_retriever = BM25Retriever(store.get_all_chunks())
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    print(f"Question: {question}\n")
    print_chunks("Dense only", dense_retriever.retrieve(question))
    print_chunks("BM25 only", [c for _, c in bm25_retriever.retrieve(question)])
    print_chunks("Hybrid (RRF-fused)", hybrid_retriever.retrieve(question))


if __name__ == "__main__":
    main()
