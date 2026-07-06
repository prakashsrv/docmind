"""Phase 4 Step 4/5: compare hybrid retrieval before and after cross-encoder
reranking for the same question -- "Top 20 -> Cross-Encoder -> Best 5" from
the project plan.

Requires `pip install sentence-transformers` -- a heavy dependency (pulls
in torch); the first run also downloads the cross-encoder model weights
from Hugging Face, so expect it to be slow once.

Usage:
    python scripts/test_rerank.py data/documents/your_file.pdf "your question"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding.vector_store import InMemoryVectorStore
from app.ingestion.pipeline import process_pdf
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_retriever import RerankingRetriever
from app.retrieval.retriever import Retriever


def print_chunks(label, chunks):
    print(f"--- {label} ({len(chunks)} chunks) ---")
    for chunk in chunks:
        print(f"  [Chunk {chunk.chunk_index}] {chunk.text[:80].strip()!r}")
    print()


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/test_rerank.py <path-to-pdf> "<question>"')
        sys.exit(1)

    path = sys.argv[1]
    question = sys.argv[2]

    _, chunks = process_pdf(path)

    store = InMemoryVectorStore()
    store.add(chunks)

    hybrid = HybridRetriever(Retriever(store), BM25Retriever(store.get_all_chunks()))
    reranked = RerankingRetriever(hybrid)

    print(f"Question: {question}\n")
    print_chunks("Hybrid only (dense + BM25, RRF-fused)", hybrid.retrieve(question, top_k=5))
    print_chunks("Hybrid + cross-encoder reranked", reranked.retrieve(question, top_k=5))


if __name__ == "__main__":
    main()
