from unittest.mock import MagicMock

from app.core import config
from app.models.chunk import Chunk
from app.retrieval.retriever import Retriever


def test_retrieve_embeds_the_question_then_searches_the_vector_store():
    fake_chunk = Chunk(
        id="1", document_id="doc", chunk_index=0, text="RAG combines retrieval with generation"
    )

    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1, 0.2, 0.3]

    vector_store = MagicMock()
    vector_store.search.return_value = [fake_chunk]

    retriever = Retriever(vector_store, embedding_service)
    result = retriever.retrieve("What is RAG?", top_k=3)

    embedding_service.embed.assert_called_once_with("What is RAG?")
    vector_store.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=3)
    assert result == [fake_chunk]


def test_retrieve_falls_back_to_config_top_k_when_not_given():
    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.0]

    vector_store = MagicMock()
    vector_store.search.return_value = []

    retriever = Retriever(vector_store, embedding_service)
    retriever.retrieve("question")

    vector_store.search.assert_called_once_with([0.0], top_k=config.TOP_K)


def test_retriever_never_touches_gemini_directly():
    # Retriever only depends on the embedding service and vector store it's
    # given -- it should have no attribute or import tying it to the LLM
    # client or ChatService.
    import app.retrieval.retriever as retriever_module

    assert not hasattr(retriever_module, "client")
    assert not hasattr(retriever_module, "ChatService")
