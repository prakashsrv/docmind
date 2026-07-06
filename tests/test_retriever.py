from unittest.mock import MagicMock

from app.core import config
from app.models.chunk import Chunk
from app.retrieval.retriever import Retriever


def make_chunk(chunk_index=0, text="RAG combines retrieval with generation"):
    return Chunk(id=str(chunk_index), document_id="doc", chunk_index=chunk_index, text=text)


def test_retrieve_embeds_the_question_then_searches_the_vector_store():
    fake_chunk = make_chunk()

    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1, 0.2, 0.3]

    vector_store = MagicMock()
    vector_store.search_with_scores.return_value = [(0.9, fake_chunk)]

    retriever = Retriever(vector_store, embedding_service)
    result = retriever.retrieve("What is RAG?", top_k=3)

    embedding_service.embed.assert_called_once_with("What is RAG?")
    vector_store.search_with_scores.assert_called_once_with([0.1, 0.2, 0.3], top_k=3)
    assert result == [fake_chunk]


def test_retrieve_falls_back_to_config_top_k_when_not_given():
    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.0]

    vector_store = MagicMock()
    vector_store.search_with_scores.return_value = []

    retriever = Retriever(vector_store, embedding_service)
    retriever.retrieve("question")

    vector_store.search_with_scores.assert_called_once_with([0.0], top_k=config.TOP_K)


def test_retrieve_drops_chunks_below_the_similarity_threshold():
    strong_match = make_chunk(chunk_index=0)
    weak_match = make_chunk(chunk_index=1)

    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1]

    vector_store = MagicMock()
    vector_store.search_with_scores.return_value = [
        (0.85, strong_match),
        (0.18, weak_match),
    ]

    retriever = Retriever(vector_store, embedding_service)
    result = retriever.retrieve("some question", min_score=0.70)

    assert result == [strong_match]


def test_retrieve_returns_nothing_when_every_candidate_is_below_threshold():
    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1]

    vector_store = MagicMock()
    vector_store.search_with_scores.return_value = [(0.18, make_chunk())]

    retriever = Retriever(vector_store, embedding_service)
    result = retriever.retrieve("who won the world cup?", min_score=0.70)

    assert result == []


def test_retrieve_uses_config_threshold_when_not_given():
    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1]

    just_above_default = config.SIMILARITY_THRESHOLD + 0.01
    just_below_default = config.SIMILARITY_THRESHOLD - 0.01
    above_chunk = make_chunk(chunk_index=0)
    below_chunk = make_chunk(chunk_index=1)

    vector_store = MagicMock()
    vector_store.search_with_scores.return_value = [
        (just_above_default, above_chunk),
        (just_below_default, below_chunk),
    ]

    retriever = Retriever(vector_store, embedding_service)
    result = retriever.retrieve("question")

    assert result == [above_chunk]


def test_retriever_never_touches_gemini_directly():
    # Retriever only depends on the embedding service and vector store it's
    # given -- it should have no attribute or import tying it to the LLM
    # client or ChatService.
    import app.retrieval.retriever as retriever_module

    assert not hasattr(retriever_module, "client")
    assert not hasattr(retriever_module, "ChatService")
