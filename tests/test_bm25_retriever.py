from app.models.chunk import Chunk
from app.retrieval.bm25_retriever import BM25Retriever


def make_chunk(chunk_index, text):
    return Chunk(id=f"c{chunk_index}", document_id="doc", chunk_index=chunk_index, text=text)


def make_chunks():
    return [
        make_chunk(0, "The backend uses FastAPI and Docker for deployment."),
        make_chunk(1, "Embeddings convert text into high dimensional vectors."),
        make_chunk(2, "Hybrid search combines BM25 keyword search with dense search."),
    ]


def test_retrieve_finds_exact_keyword_match():
    retriever = BM25Retriever(make_chunks())

    results = retriever.retrieve("FastAPI")

    assert len(results) >= 1
    assert results[0][1].chunk_index == 0
    assert results[0][0] > 0


def test_retrieve_excludes_chunks_with_zero_term_overlap():
    retriever = BM25Retriever(make_chunks())

    results = retriever.retrieve("FastAPI")
    matched_indexes = {chunk.chunk_index for _, chunk in results}

    # Chunk 1 (embeddings) and chunk 2 (hybrid search) share no terms with
    # "FastAPI" and shouldn't be returned just to fill out top_k.
    assert 1 not in matched_indexes
    assert 2 not in matched_indexes


def test_retrieve_returns_empty_list_for_completely_unrelated_query():
    retriever = BM25Retriever(make_chunks())

    results = retriever.retrieve("a completely unrelated query about cooking recipes")

    assert results == []


def test_retrieve_handles_an_empty_corpus():
    retriever = BM25Retriever([])

    assert retriever.retrieve("anything") == []


def test_retrieve_respects_top_k():
    # A term that appears in every single document has degenerate (negative)
    # IDF in BM25's formula, so mix in some chunks that don't mention
    # "search" at all -- otherwise every score comes back <= 0 and the
    # zero-overlap filter would (correctly) drop everything.
    matching = [make_chunk(i, f"search result number {i}") for i in range(10)]
    decoys = [make_chunk(100 + i, f"unrelated filler content {i}") for i in range(3)]
    retriever = BM25Retriever(matching + decoys)

    results = retriever.retrieve("search", top_k=3)

    assert len(results) == 3
