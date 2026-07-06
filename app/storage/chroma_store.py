import json

import chromadb

from app.core import config
from app.models.chunk import Chunk


class ChromaVectorStore:
    """Same add() / search() / search_with_scores() interface as
    InMemoryVectorStore, backed by a persistent ChromaDB collection on disk
    instead of a Python list.

    Retriever only ever calls those three methods, so swapping this in for
    InMemoryVectorStore elsewhere in the app is meant to be close to a
    one-line change -- e.g. `Retriever(ChromaVectorStore())` instead of
    `Retriever(InMemoryVectorStore())`.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        collection_name = collection_name or config.CHROMA_COLLECTION_NAME

        self.client = chromadb.PersistentClient(path=persist_dir)
        # hnsw:space="cosine" makes Chroma report *cosine distance*
        # (1 - cosine_similarity) rather than its default squared L2 --
        # matching the 0-1 similarity scale the rest of the app already
        # assumes (SIMILARITY_THRESHOLD, cosine_similarity() in
        # vector_store.py).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks by id. Since chunk ids are derived deterministically
        from document content (see chunker.py), re-ingesting the same PDF
        overwrites its existing rows instead of duplicating them.
        """
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "metadata_json": json.dumps(chunk.metadata or {}),
                }
                for chunk in chunks
            ],
        )

    def search_with_scores(
        self, query_vector: list[float], top_k: int = None
    ) -> list[tuple[float, Chunk]]:
        top_k = top_k or config.TOP_K
        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        scored = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma returns cosine *distance*; the rest of the app deals in
            # cosine *similarity* (1.0 = identical), so convert here once
            # rather than leaking Chroma's convention into Retriever/RagService.
            similarity = 1 - distance
            chunk = Chunk(
                id=chunk_id,
                document_id=metadata["document_id"],
                chunk_index=metadata["chunk_index"],
                text=text,
                embedding=None,  # not requested back -- see note in docs
                metadata=json.loads(metadata.get("metadata_json", "{}")),
            )
            scored.append((similarity, chunk))

        return scored

    def search(self, query_vector: list[float], top_k: int = None) -> list[Chunk]:
        return [chunk for _, chunk in self.search_with_scores(query_vector, top_k=top_k)]
