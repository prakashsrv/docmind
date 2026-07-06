from app.core import config
from app.models.chunk import Chunk

# Loading a cross-encoder is a real cost: model weights, several seconds,
# meaningful memory -- and `sentence-transformers` pulls in torch, easily
# the heaviest dependency in this project. Both the import and the model
# load are deferred to first use inside _get_model(), rather than done at
# module level, so importing app.retrieval.reranker (or anything that
# depends on it, like RerankingRetriever) doesn't require torch to be
# installed at all unless reranking is actually invoked -- which matters
# for every other retriever's tests, and for anyone who hasn't installed
# this optional dependency yet.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _model


class CrossEncoderReranker:
    """Chunks -> re-scored, re-ordered Chunks, using a cross-encoder model.

    Dense search and BM25 both encode the question and each chunk
    *separately* (as an embedding vector, or as term-frequency statistics)
    and compare those fixed representations -- fast, because every chunk's
    representation can be precomputed once and reused across every future
    query. A cross-encoder instead feeds the question and a single
    candidate chunk into the model *together*, in one pass, and directly
    predicts a relevance score for that specific pairing. That tends to be
    noticeably more accurate, because the model can actually attend to how
    the two texts relate to each other rather than comparing two vectors
    computed in isolation -- but there's no way to precompute anything, so
    it doesn't scale to searching an entire corpus. Hence: rerank only a
    modest shortlist (the base retriever's top-N), never the full chunk set.
    """

    def rerank(self, question: str, chunks: list[Chunk], top_k: int = None) -> list[Chunk]:
        top_k = top_k or config.TOP_K

        if not chunks:
            return []

        pairs = [(question, chunk.text) for chunk in chunks]
        scores = _get_model().predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
