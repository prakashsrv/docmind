from unittest.mock import MagicMock, patch

from app.models.chunk import Chunk
from app.retrieval.reranker import CrossEncoderReranker


def make_chunk(chunk_index):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=str(chunk_index))


@patch("app.retrieval.reranker._get_model")
def test_rerank_reorders_by_model_score_not_input_order(mock_get_model):
    a, b, c = make_chunk(0), make_chunk(1), make_chunk(2)

    mock_model = MagicMock()
    # Input order is [a, b, c]; the model says b is actually most relevant.
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    mock_get_model.return_value = mock_model

    result = CrossEncoderReranker().rerank("question", [a, b, c])

    assert [chunk.id for chunk in result] == [b.id, c.id, a.id]


@patch("app.retrieval.reranker._get_model")
def test_rerank_scores_question_and_chunk_text_as_pairs(mock_get_model):
    chunk = make_chunk(0)
    chunk.text = "the backend uses FastAPI"

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8]
    mock_get_model.return_value = mock_model

    CrossEncoderReranker().rerank("what does the backend use?", [chunk])

    mock_model.predict.assert_called_once_with(
        [("what does the backend use?", "the backend uses FastAPI")]
    )


@patch("app.retrieval.reranker._get_model")
def test_rerank_respects_top_k(mock_get_model):
    chunks = [make_chunk(i) for i in range(5)]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_get_model.return_value = mock_model

    result = CrossEncoderReranker().rerank("question", chunks, top_k=2)

    assert len(result) == 2


@patch("app.retrieval.reranker._get_model")
def test_rerank_of_empty_chunk_list_skips_the_model_entirely(mock_get_model):
    result = CrossEncoderReranker().rerank("question", [])

    assert result == []
    mock_get_model.assert_not_called()
