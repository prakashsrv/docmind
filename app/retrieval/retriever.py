import logging

from app.core import config
from app.embedding.embedding_service import EmbeddingService
from app.embedding.vector_store import InMemoryVectorStore
from app.models.chunk import Chunk

logger = logging.getLogger("docmind")


class Retriever:
    """Question -> relevant Chunks.

    Knows how to embed a question and search a vector store. Deliberately
    never talks to Gemini for generation -- that responsibility belongs to
    RagService, so this class can be tested, swapped, or reused (e.g. for
    hybrid search later) without touching anything about how answers get
    generated.
    """

    def __init__(self, vector_store: InMemoryVectorStore, embedding_service: EmbeddingService = None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or EmbeddingService()

    def retrieve(
        self,
        question: str,
        top_k: int = None,
        min_score: float = None,
    ) -> list[Chunk]:
        """Embed the question, search the vector store, and only keep
        chunks that clear a minimum similarity score.

        Vector search on its own always returns *something* -- the top-k
        least-bad matches, even for a question with no relevant content in
        the store at all. The threshold is what lets this method return
        nothing rather than confidently-wrong context, which is what makes
        the "I couldn't find that information" fallback in RagService
        actually reliable instead of dependent on the LLM's good behavior.
        """
        top_k = top_k or config.TOP_K
        min_score = config.SIMILARITY_THRESHOLD if min_score is None else min_score

        query_vector = self.embedding_service.embed(question)
        scored = self.vector_store.search_with_scores(query_vector, top_k=top_k)

        best_score = scored[0][0] if scored else None
        logger.info(
            "Retrieval for %r: best_score=%s threshold=%.2f candidates=%d",
            question,
            f"{best_score:.3f}" if best_score is not None else "n/a",
            min_score,
            len(scored),
        )

        accepted = [(score, chunk) for score, chunk in scored if score >= min_score]

        if len(accepted) < len(scored):
            logger.info(
                "Rejected %d/%d candidates below threshold %.2f",
                len(scored) - len(accepted),
                len(scored),
                min_score,
            )

        return [chunk for _, chunk in accepted]
