from app.core import config
from app.models.chunk import Chunk
from app.retrieval.reranker import CrossEncoderReranker


class RerankingRetriever:
    """Question -> relevant Chunks, adding a cross-encoder reranking pass on
    top of any other retriever (Retriever, HybridRetriever, ...).

    This is "Top 20 -> Cross-Encoder -> Best 5" from the project plan: pull
    a wider candidate pool from whatever retriever it wraps, then let the
    reranker narrow that down to top_k using pairwise question+chunk
    scoring instead of whatever ranking the base retriever produced.

    Exposes the same retrieve(question, top_k) -> list[Chunk] signature as
    every other retriever in this package, so it's a drop-in replacement
    for RagService regardless of what it wraps -- e.g.
    RagService(RerankingRetriever(HybridRetriever(dense, bm25))).
    """

    def __init__(
        self,
        base_retriever,
        reranker: CrossEncoderReranker = None,
        candidate_pool: int = None,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_pool = candidate_pool or config.RERANK_CANDIDATE_POOL

    def retrieve(self, question: str, top_k: int = None) -> list[Chunk]:
        top_k = top_k or config.TOP_K

        # The base retriever's own relevance gating (similarity threshold,
        # BM25 zero-overlap filter, or both via HybridRetriever) still
        # applies here -- reranking only re-orders candidates the base
        # retriever already considered plausible, it doesn't independently
        # decide "is there anything relevant in this corpus at all." If the
        # base retriever finds nothing, there's nothing to rerank, and the
        # "I don't know" guarantee passes through unchanged.
        candidates = self.base_retriever.retrieve(question, top_k=self.candidate_pool)

        if not candidates:
            return []

        return self.reranker.rerank(question, candidates, top_k=top_k)
