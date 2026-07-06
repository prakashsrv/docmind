from unittest.mock import MagicMock, patch

from app.llm.prompts import NOT_FOUND_MESSAGE
from app.models.chunk import Chunk
from evaluation.evaluator import _judge, _retrieval_hit, evaluate, summarize


def make_chunk(chunk_index, text):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=text)


# -- _retrieval_hit -----------------------------------------------------


def test_retrieval_hit_true_when_all_keywords_present_across_chunks():
    assert _retrieval_hit(["fastapi", "docker"], ["the backend uses FastAPI", "and also Docker"])


def test_retrieval_hit_false_when_a_keyword_is_missing():
    assert not _retrieval_hit(["fastapi", "docker"], ["the backend uses FastAPI"])


def test_retrieval_hit_false_when_no_keywords_were_specified():
    # An empty keyword list shouldn't count as a vacuous "hit".
    assert not _retrieval_hit([], ["anything at all"])


# -- _judge ---------------------------------------------------------------


@patch("evaluation.evaluator.complete")
def test_judge_parses_a_clean_json_response(mock_complete):
    mock_complete.return_value = '{"faithfulness": 0.9, "relevance": 0.8}'

    faithfulness, relevance = _judge("q", "context", "answer")

    assert faithfulness == 0.9
    assert relevance == 0.8


@patch("evaluation.evaluator.complete")
def test_judge_strips_markdown_code_fences(mock_complete):
    mock_complete.return_value = '```json\n{"faithfulness": 1.0, "relevance": 1.0}\n```'

    faithfulness, relevance = _judge("q", "context", "answer")

    assert faithfulness == 1.0
    assert relevance == 1.0


@patch("evaluation.evaluator.complete")
def test_judge_scores_zero_on_unparseable_response_instead_of_raising(mock_complete):
    mock_complete.return_value = "I think this answer looks pretty good actually."

    faithfulness, relevance = _judge("q", "context", "answer")

    assert (faithfulness, relevance) == (0.0, 0.0)


# -- evaluate / summarize -------------------------------------------------


@patch("evaluation.evaluator._judge")
def test_evaluate_scores_answerable_question_with_retrieval_hit_and_judge(mock_judge):
    mock_judge.return_value = (0.95, 0.9)

    chunk = make_chunk(1, "The backend uses FastAPI and Docker.")
    rag_service = MagicMock()
    rag_service.answer_with_chunks.return_value = ("FastAPI and Docker.\n\nSources: Chunk 1", [chunk])

    dataset = [
        {"id": "q1", "question": "What does the backend use?", "answerable": True,
         "expected_chunk_keywords": ["FastAPI", "Docker"]}
    ]

    results = evaluate(rag_service, dataset)

    assert len(results) == 1
    assert results[0].retrieval_hit is True
    assert results[0].faithfulness == 0.95
    assert results[0].relevance == 0.9
    assert results[0].correctly_rejected is None


def test_evaluate_scores_unanswerable_question_by_exact_rejection_match():
    rag_service = MagicMock()
    rag_service.answer_with_chunks.return_value = (NOT_FOUND_MESSAGE, [])

    dataset = [{"id": "n1", "question": "Who won the world cup?", "answerable": False}]

    results = evaluate(rag_service, dataset)

    assert results[0].correctly_rejected is True
    assert results[0].retrieval_hit is None
    assert results[0].faithfulness is None


def test_evaluate_marks_unanswerable_question_wrong_if_system_answered_anyway():
    rag_service = MagicMock()
    rag_service.answer_with_chunks.return_value = ("The world cup was won by Argentina.", [])

    dataset = [{"id": "n1", "question": "Who won the world cup?", "answerable": False}]

    results = evaluate(rag_service, dataset)

    assert results[0].correctly_rejected is False


def test_summarize_aggregates_across_answerable_and_unanswerable_questions():
    from evaluation.evaluator import EvalResult

    results = [
        EvalResult("q1", "q", True, "a", [0], retrieval_hit=True, faithfulness=1.0, relevance=1.0),
        EvalResult("q2", "q", True, "a", [1], retrieval_hit=False, faithfulness=0.0, relevance=0.5),
        EvalResult("n1", "q", False, "a", [], correctly_rejected=True),
        EvalResult("n2", "q", False, "a", [], correctly_rejected=False),
    ]

    summary = summarize(results)

    assert summary["num_questions"] == 4
    assert summary["num_answerable"] == 2
    assert summary["num_unanswerable"] == 2
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["faithfulness"] == 0.5
    assert summary["relevance"] == 0.75
    assert summary["correct_rejection_rate"] == 0.5
