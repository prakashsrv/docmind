from unittest.mock import MagicMock

from app.models.chunk import Chunk
from app.retrieval.hybrid_retriever import HybridRetriever, _reciprocal_rank_fusion


def make_chunk(chunk_index):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=str(chunk_index))


def test_rrf_ranks_a_chunk_found_by_both_lists_above_one_found_by_only_one():
    a, b, c = make_chunk(0), make_chunk(1), make_chunk(2)

    # `a` is #1 in both lists; `b` is #2 in the dense list only.
    dense = [a, b]
    bm25 = [a, c]

    fused = _reciprocal_rank_fusion([dense, bm25])

    assert fused[0].id == a.id


def test_rrf_includes_chunks_that_appear_in_only_one_list():
    a, b = make_chunk(0), make_chunk(1)

    fused = _reciprocal_rank_fusion([[a], [b]])

    assert {chunk.id for chunk in fused} == {a.id, b.id}


def test_rrf_of_empty_lists_is_empty():
    assert _reciprocal_rank_fusion([[], []]) == []


def test_hybrid_retriever_returns_nothing_when_both_sources_are_empty():
    dense_retriever = MagicMock()
    dense_retriever.retrieve.return_value = []

    bm25_retriever = MagicMock()
    bm25_retriever.retrieve.return_value = []

    hybrid = HybridRetriever(dense_retriever, bm25_retriever)
    result = hybrid.retrieve("who won the world cup?")

    assert result == []


def test_hybrid_retriever_falls_back_to_bm25_when_dense_finds_nothing():
    # This is the scenario hybrid retrieval exists for: an exact keyword
    # match (e.g. a library name) that the embedding-based Retriever
    # rejected via its similarity threshold, but BM25 catches directly.
    exact_match = make_chunk(0)

    dense_retriever = MagicMock()
    dense_retriever.retrieve.return_value = []  # nothing cleared the threshold

    bm25_retriever = MagicMock()
    bm25_retriever.retrieve.return_value = [(4.2, exact_match)]

    hybrid = HybridRetriever(dense_retriever, bm25_retriever)
    result = hybrid.retrieve("PyMuPDF")

    assert result == [exact_match]


def test_hybrid_retriever_respects_top_k_after_fusion():
    chunks = [make_chunk(i) for i in range(10)]

    dense_retriever = MagicMock()
    dense_retriever.retrieve.return_value = chunks

    bm25_retriever = MagicMock()
    bm25_retriever.retrieve.return_value = []

    hybrid = HybridRetriever(dense_retriever, bm25_retriever)
    result = hybrid.retrieve("question", top_k=3)

    assert len(result) == 3
