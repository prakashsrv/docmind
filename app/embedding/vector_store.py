import math

from app.core import config
from app.models.chunk import Chunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = identical direction (same meaning), 0 = unrelated, -1 = opposite.

    Two vectors point in a similar direction when their dot product is high
    relative to their lengths -- dividing by both norms cancels out vector
    length so we're comparing direction only, not magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """The simplest possible vector database: a Python list and a linear
    scan. It's O(n) per search, which is fine for hundreds of chunks and
    exactly what Chroma (or any real vector DB) optimizes away later with
    indexing -- same `add`/`search` interface, smarter internals.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query_vector: list[float], top_k: int = None) -> list[Chunk]:
        top_k = top_k or config.TOP_K

        scored = [
            (cosine_similarity(query_vector, chunk.embedding), chunk)
            for chunk in self.chunks
            if chunk.embedding is not None
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [chunk for _, chunk in scored[:top_k]]
