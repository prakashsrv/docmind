from unittest.mock import MagicMock, patch

from app.llm.prompts import NOT_FOUND_MESSAGE
from app.models.chunk import Chunk
from app.retrieval.rag_service import RagService


def make_chunk(chunk_index, text="some retrieved text"):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=text)


def test_answer_returns_not_found_message_when_retriever_finds_nothing():
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    result = RagService(retriever).answer("who won the world cup?")

    assert result == NOT_FOUND_MESSAGE


@patch("app.retrieval.rag_service.complete")
def test_answer_appends_deterministic_sources_from_retrieved_chunks(mock_complete):
    chunks = [make_chunk(2), make_chunk(5)]
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    mock_complete.return_value = "FastAPI and Docker are used."

    result = RagService(retriever).answer("what does the backend use?")

    assert result == "FastAPI and Docker are used.\n\nSources: Chunk 2, Chunk 5"


@patch("app.retrieval.rag_service.complete")
def test_answer_does_not_attach_sources_when_model_says_not_found(mock_complete):
    retriever = MagicMock()
    retriever.retrieve.return_value = [make_chunk(0)]
    mock_complete.return_value = NOT_FOUND_MESSAGE

    result = RagService(retriever).answer("some question")

    assert result == NOT_FOUND_MESSAGE
    assert "Sources:" not in result


@patch("app.retrieval.rag_service.complete")
def test_answer_with_chunks_returns_both_the_answer_and_the_chunks_used(mock_complete):
    chunks = [make_chunk(1)]
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    mock_complete.return_value = "The answer."

    answer, returned_chunks = RagService(retriever).answer_with_chunks("question")

    assert answer == "The answer.\n\nSources: Chunk 1"
    assert returned_chunks == chunks


@patch("app.retrieval.rag_service.complete")
def test_answer_with_chunks_only_calls_the_retriever_once(mock_complete):
    # This is the whole point of answer_with_chunks existing: callers that
    # need both the answer and the chunks (like the eval harness) shouldn't
    # have to retrieve twice -- once implicitly inside answer(), once again
    # themselves -- which would double the embedding API calls per question.
    retriever = MagicMock()
    retriever.retrieve.return_value = [make_chunk(0)]
    mock_complete.return_value = "an answer"

    RagService(retriever).answer_with_chunks("question")

    retriever.retrieve.assert_called_once()


def test_answer_with_chunks_returns_empty_chunk_list_when_nothing_retrieved():
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    answer, chunks = RagService(retriever).answer_with_chunks("question")

    assert answer == NOT_FOUND_MESSAGE
    assert chunks == []
