from app.core import config
from app.llm.client import client
from app.models.chunk import Chunk


class EmbeddingService:
    """One responsibility: text -> vector. Nothing here knows about PDFs,
    chunking, or storage -- just Gemini's embedding model.
    """

    def embed(self, text: str) -> list[float]:
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL_NAME,
            contents=text,
        )
        return response.embeddings[0].values

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed every chunk in a single batched API call rather than one
        request per chunk -- fewer round trips, and the API is built to
        accept a list of texts at once.
        """
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL_NAME,
            contents=texts,
        )

        for chunk, embedding in zip(chunks, response.embeddings):
            chunk.embedding = embedding.values

        return chunks
