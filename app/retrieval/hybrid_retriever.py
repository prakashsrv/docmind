from app.core import config
from app.models.chunk import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.retriever import Retriever

# Standard smoothing constant for reciprocal rank fusion. Larger values
# flatten the difference between rank 1 and rank 10; 60 is the value used
# in the original RRF paper and most hybrid-search implementations that
# followed it -- there's nothing DocMind-specific about it.
RRF_K = 60


def _reciprocal_rank_fusion(ranked_lists: list[list[Chunk]], k: int = RRF_K) -> list[Chunk]:
    """Merge multiple ranked chunk lists into one, using each chunk's
    *rank position* rather than its raw score.

    Dense (cosine similarity, 0-1) and BM25 (unbounded, corpus-dependent)
    scores live on completely different scales, so they can't be averaged
    or compared directly -- a BM25 score of 8.3 says nothing about whether
    it's better or worse than a cosine similarity of 0.85. RRF sidesteps
    that by ignoring raw scores entirely: a chunk ranked #1 by both methods
    clearly deserves to beat one ranked #1 by only one of them, and that's
    true regardless of what scale either method's scores are on.

    fused_score(chunk) = sum over each list it appears in of 1 / (k + rank)
    """
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
    return [chunks_by_id[chunk_id] for chunk_id in ranked_ids]


class HybridRetriever:
    """Question -> relevant Chunks, combining dense (embedding) search and
    BM25 (keyword) search.

    Each underlying retriever contributes only the chunks *it's* confident
    about -- Retriever's own similarity threshold, BM25Retriever's own
    nonzero-overlap filter -- so a chunk only shows up here if at least one
    retrieval method actually found it relevant, not just "the fifth-best
    guess when nothing was good." If both underlying retrievers come back
    empty, this returns [] too, which still triggers RagService's
    NOT_FOUND_MESSAGE the same way a plain Retriever would.

    Exposes the same retrieve(question, top_k) -> list[Chunk] signature as
    Retriever, so it's a drop-in replacement anywhere a Retriever is used
    (e.g. RagService).
    """

    def __init__(self, dense_retriever: Retriever, bm25_retriever: BM25Retriever):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, question: str, top_k: int = None) -> list[Chunk]:
        top_k = top_k or config.TOP_K
        # Look wider than top_k on each side before fusing -- a chunk
        # ranked #8 by dense and #2 by BM25 might still deserve a top-5
        # spot in the fused list, but only if we gave it a chance to show
        # up in both candidate lists in the first place.
        candidate_pool = top_k * 4

        dense_candidates = self.dense_retriever.retrieve(question, top_k=candidate_pool)
        bm25_candidates = [
            chunk for _, chunk in self.bm25_retriever.retrieve(question, top_k=candidate_pool)
        ]

        if not dense_candidates and not bm25_candidates:
            return []

        fused = _reciprocal_rank_fusion([dense_candidates, bm25_candidates])
        return fused[:top_k]
