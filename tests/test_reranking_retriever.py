from unittest.mock import MagicMock

from app.models.chunk import Chunk
from app.retrieval.reranking_retriever import RerankingRetriever


def make_chunk(chunk_index):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=str(chunk_index))


def test_retrieve_pulls_a_wider_candidate_pool_than_top_k():
    base_retriever = MagicMock()
    base_retriever.retrieve.return_value = []

    reranker = MagicMock()
    reranker.rerank.return_value = []

    retriever = RerankingRetriever(base_retriever, reranker, candidate_pool=20)
    retriever.retrieve("question", top_k=5)

    base_retriever.retrieve.assert_called_once_with("question", top_k=20)


def test_retrieve_passes_base_retrievers_candidates_to_the_reranker():
    candidates = [make_chunk(i) for i in range(3)]
    reranked = [candidates[2], candidates[0]]

    base_retriever = MagicMock()
    base_retriever.retrieve.return_value = candidates

    reranker = MagicMock()
    reranker.rerank.return_value = reranked

    retriever = RerankingRetriever(base_retriever, reranker)
    result = retriever.retrieve("question", top_k=2)

    reranker.rerank.assert_called_once_with("question", candidates, top_k=2)
    assert result == reranked


def test_retrieve_skips_reranking_entirely_when_base_retriever_finds_nothing():
    base_retriever = MagicMock()
    base_retriever.retrieve.return_value = []

    reranker = MagicMock()

    retriever = RerankingRetriever(base_retriever, reranker)
    result = retriever.retrieve("who won the world cup?")

    assert result == []
    reranker.rerank.assert_not_called()
