"""Stage 6: the evaluation harness.

Runs a golden dataset of question/answer pairs through a RagService and
measures two separate things per the project plan:

- Retrieval quality: did the retriever actually fetch a chunk containing
  the information needed to answer the question? Approximated here as
  "does the concatenated retrieved context contain all of this question's
  expected keywords" -- a cheap proxy for "we don't have a hand-labeled
  correct chunk index for every question," not full retrieval-relevance
  scoring like RAGAS's context_precision/context_recall.
- Faithfulness and relevance: scored by an LLM-as-a-judge (Gemini itself,
  via a separate prompt) rather than RAGAS, to avoid adding another heavy
  dependency -- same idea, hand-rolled.

For unanswerable questions (no real answer exists in the documents), the
only thing that matters is whether the system correctly said so via
NOT_FOUND_MESSAGE -- a deterministic string comparison, no judge needed.
"""

import json
from dataclasses import dataclass

from app.llm.chat_service import complete
from app.llm.prompts import NOT_FOUND_MESSAGE
from app.retrieval.rag_service import RagService

JUDGE_PROMPT_TEMPLATE = """You are grading an AI system's answer for a RAG evaluation benchmark.

Question:
{question}

Retrieved context the system was given:
{context}

System's answer:
{answer}

Score the answer on two dimensions, each a float from 0 to 1:
- faithfulness: does the answer only state things actually supported by
  the retrieved context above? 1.0 = fully grounded, nothing invented.
  0.0 = mostly or entirely unsupported by the context.
- relevance: does the answer actually address the question asked?
  1.0 = directly and completely answers it. 0.0 = off-topic or non-responsive.

Respond with ONLY a JSON object and nothing else, in exactly this form:
{{"faithfulness": 0.0, "relevance": 0.0}}
"""


@dataclass
class EvalResult:
    question_id: str
    question: str
    answerable: bool
    answer: str
    retrieved_chunk_indexes: list[int]
    retrieval_hit: bool | None = None       # only meaningful for answerable questions
    correctly_rejected: bool | None = None  # only meaningful for unanswerable questions
    faithfulness: float | None = None
    relevance: float | None = None


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _judge(question: str, context: str, answer: str) -> tuple[float, float]:
    """LLM-as-a-judge: ask Gemini to score faithfulness and relevance of an
    answer against the context it was actually given. Returns (0.0, 0.0)
    if the judge's response can't be parsed, rather than raising -- a
    single malformed judge response shouldn't crash an entire eval run
    over a 20+ question dataset; it just scores that one question as a
    failure, which is visible in the results either way.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)
    raw = complete(prompt)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return float(data["faithfulness"]), float(data["relevance"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0, 0.0


def _retrieval_hit(expected_keywords: list[str], retrieved_texts: list[str]) -> bool:
    if not expected_keywords:
        return False

    combined = " ".join(retrieved_texts).lower()
    return all(keyword.lower() in combined for keyword in expected_keywords)


def evaluate(rag_service: RagService, dataset: list[dict]) -> list[EvalResult]:
    """Run every question in the dataset through rag_service and score it."""
    results = []

    for item in dataset:
        question = item["question"]
        answerable = item.get("answerable", True)

        answer, chunks = rag_service.answer_with_chunks(question)
        retrieved_indexes = [chunk.chunk_index for chunk in chunks]

        if answerable:
            hit = _retrieval_hit(item.get("expected_chunk_keywords", []), [c.text for c in chunks])
            context = "\n\n".join(chunk.text for chunk in chunks) if chunks else "(no context retrieved)"
            faithfulness, relevance = _judge(question, context, answer)

            results.append(
                EvalResult(
                    question_id=item["id"],
                    question=question,
                    answerable=True,
                    answer=answer,
                    retrieved_chunk_indexes=retrieved_indexes,
                    retrieval_hit=hit,
                    faithfulness=faithfulness,
                    relevance=relevance,
                )
            )
        else:
            correctly_rejected = answer.strip() == NOT_FOUND_MESSAGE
            results.append(
                EvalResult(
                    question_id=item["id"],
                    question=question,
                    answerable=False,
                    answer=answer,
                    retrieved_chunk_indexes=retrieved_indexes,
                    correctly_rejected=correctly_rejected,
                )
            )

    return results


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate a list of EvalResults into the headline numbers."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def rate(flags: list[bool]) -> float | None:
        return sum(1 for f in flags if f) / len(flags) if flags else None

    def avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "num_questions": len(results),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "retrieval_hit_rate": rate([r.retrieval_hit for r in answerable]),
        "faithfulness": avg([r.faithfulness for r in answerable if r.faithfulness is not None]),
        "relevance": avg([r.relevance for r in answerable if r.relevance is not None]),
        "correct_rejection_rate": rate([r.correctly_rejected for r in unanswerable]),
    }
