from app.core import config
from app.embedding.embedding_service import EmbeddingService
from app.embedding.vector_store import InMemoryVectorStore
from app.models.chunk import Chunk


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

    def retrieve(self, question: str, top_k: int = None) -> list[Chunk]:
        top_k = top_k or config.TOP_K
        query_vector = self.embedding_service.embed(question)
        return self.vector_store.search(query_vector, top_k=top_k)
