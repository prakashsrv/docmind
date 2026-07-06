from unittest.mock import MagicMock, patch

from app.embedding.embedding_service import EmbeddingService
from app.models.chunk import Chunk


def make_fake_response(vectors):
    """Build an object that looks like google-genai's EmbedContentResponse
    without hitting the real API -- vectors is a list of lists of floats.
    """
    fake_embeddings = [MagicMock(values=vector) for vector in vectors]
    return MagicMock(embeddings=fake_embeddings)


@patch("app.embedding.embedding_service.client")
def test_embed_returns_first_embedding_values(mock_client):
    mock_client.models.embed_content.return_value = make_fake_response([[0.1, 0.2, 0.3]])

    result = EmbeddingService().embed("hello")

    assert result == [0.1, 0.2, 0.3]
    mock_client.models.embed_content.assert_called_once()


@patch("app.embedding.embedding_service.client")
def test_embed_chunks_assigns_one_embedding_per_chunk_via_a_single_call(mock_client):
    chunks = [
        Chunk(id="1", document_id="doc", chunk_index=0, text="a"),
        Chunk(id="2", document_id="doc", chunk_index=1, text="b"),
    ]
    mock_client.models.embed_content.return_value = make_fake_response(
        [[1.0, 0.0], [0.0, 1.0]]
    )

    result = EmbeddingService().embed_chunks(chunks)

    assert result[0].embedding == [1.0, 0.0]
    assert result[1].embedding == [0.0, 1.0]
    # The whole point of embed_chunks is one batched call, not N calls.
    assert mock_client.models.embed_content.call_count == 1


@patch("app.embedding.embedding_service.client")
def test_embed_chunks_skips_the_api_entirely_for_an_empty_list(mock_client):
    result = EmbeddingService().embed_chunks([])

    assert result == []
    mock_client.models.embed_content.assert_not_called()
