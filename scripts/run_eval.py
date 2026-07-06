"""Stage 6 evaluation harness runner: run the golden dataset through one or
more retrieval configurations and print retrieval-hit-rate / faithfulness /
relevance / correct-rejection-rate for each -- this is what turns "hybrid +
reranking should be better" into an actual measured before/after number.

Note: this calls the real Gemini API for every question -- once to
generate the answer, once more for the LLM-as-a-judge scoring of each
answerable question -- so it costs real tokens and takes real time,
proportional to len(dataset) * len(modes). The default dataset has 22
questions; running all three modes means roughly 22 * 3 = 66 answer calls
plus ~48 judge calls (16 answerable questions x 3 modes).

The Gemini API free tier caps requests at 5 per minute (as of writing),
which this call volume blows past immediately -- chat_service.complete()
retries on 429s using the server's suggested delay, so a free-tier run
will *succeed*, just much more slowly than the raw call count suggests
(expect low tens of minutes for a full 3-mode run, not seconds). Use
--limit to sanity-check on a handful of questions first, and --modes to
run one configuration at a time.

Usage:
    python scripts/run_eval.py data/documents/your_file.pdf
    python scripts/run_eval.py data/documents/your_file.pdf --modes dense
    python scripts/run_eval.py data/documents/your_file.pdf --limit 5
    python scripts/run_eval.py data/documents/your_file.pdf --dataset evaluation/golden_dataset.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding.vector_store import InMemoryVectorStore
from app.ingestion.pipeline import process_pdf
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rag_service import RagService
from app.retrieval.retriever import Retriever
from evaluation.evaluator import evaluate, load_golden_dataset, summarize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "evaluation", "golden_dataset.json")


def build_retriever(mode: str, store: InMemoryVectorStore):
    dense = Retriever(store)

    if mode == "dense":
        return dense

    if mode == "hybrid":
        return HybridRetriever(dense, BM25Retriever(store.get_all_chunks()))

    if mode == "hybrid_rerank":
        # Imported lazily -- requires `pip install sentence-transformers`,
        # which the other two modes don't need at all.
        from app.retrieval.reranking_retriever import RerankingRetriever

        return RerankingRetriever(HybridRetriever(dense, BM25Retriever(store.get_all_chunks())))

    raise ValueError(f"Unknown mode: {mode!r} (expected dense, hybrid, or hybrid_rerank)")


def fmt(value):
    return f"{value:.2f}" if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="PDF to ingest and evaluate against")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to a golden dataset JSON file")
    parser.add_argument(
        "--modes",
        default="dense,hybrid,hybrid_rerank",
        help="Comma-separated: dense, hybrid, hybrid_rerank",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions -- useful for a quick, low-quota smoke test",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    dataset = load_golden_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    print(f"Golden dataset: {len(dataset)} questions ({args.dataset})")
    print(f"Ingesting: {args.pdf_path}\n")

    _, chunks = process_pdf(args.pdf_path)
    store = InMemoryVectorStore()
    store.add(chunks)

    rows = []
    for mode in modes:
        print(f"Running mode: {mode} ...")
        start = time.perf_counter()

        try:
            retriever = build_retriever(mode, store)
        except ImportError as e:
            print(f"  skipped -- {e}\n")
            continue

        rag_service = RagService(retriever)
        results = evaluate(rag_service, dataset)
        summary = summarize(results)
        summary["mode"] = mode
        summary["elapsed_seconds"] = round(time.perf_counter() - start, 1)
        rows.append(summary)

        print(f"  done in {summary['elapsed_seconds']}s\n")

    if not rows:
        print("No modes ran successfully.")
        return

    print(f"{'Mode':<15}{'Retrieval hit':<15}{'Faithfulness':<15}{'Relevance':<12}{'Reject rate':<13}{'Time (s)'}")
    for row in rows:
        print(
            f"{row['mode']:<15}"
            f"{fmt(row['retrieval_hit_rate']):<15}"
            f"{fmt(row['faithfulness']):<15}"
            f"{fmt(row['relevance']):<12}"
            f"{fmt(row['correct_rejection_rate']):<13}"
            f"{row['elapsed_seconds']}"
        )


if __name__ == "__main__":
    main()
