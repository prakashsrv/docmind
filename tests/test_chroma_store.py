from app.models.chunk import Chunk
from app.storage.chroma_store import ChromaVectorStore

# Chroma is a real embedded database (no network calls), so these are
# genuine integration tests against a throwaway on-disk store per test
# (pytest's tmp_path fixture), rather than mocks.


def make_store(tmp_path, name="test-collection"):
    return ChromaVectorStore(persist_dir=str(tmp_path), collection_name=name)


def test_add_then_search_with_scores_ranks_best_match_first(tmp_path):
    store = make_store(tmp_path)
    close_match = Chunk(
        id="doc-0", document_id="doc", chunk_index=0, text="close", embedding=[1.0, 0.0, 0.0]
    )
    far_match = Chunk(
        id="doc-1", document_id="doc", chunk_index=1, text="far", embedding=[0.0, 1.0, 0.0]
    )
    store.add([far_match, close_match])

    results = store.search_with_scores([1.0, 0.0, 0.0], top_k=2)

    assert [chunk.id for _, chunk in results] == ["doc-0", "doc-1"]
    assert results[0][0] > results[1][0]
    assert results[0][0] == 1.0  # identical vector -> cosine similarity 1.0


def test_search_returns_empty_list_for_empty_store(tmp_path):
    store = make_store(tmp_path)

    assert store.search_with_scores([1.0, 0.0]) == []
    assert store.search([1.0, 0.0]) == []


def test_add_upserts_by_id_instead_of_duplicating(tmp_path):
    store = make_store(tmp_path)
    chunk = Chunk(id="doc-0", document_id="doc", chunk_index=0, text="v1", embedding=[1.0, 0.0])

    store.add([chunk])
    store.add([chunk])  # simulate re-ingesting the same document

    assert store.collection.count() == 1


def test_search_reconstructs_chunk_fields_from_metadata(tmp_path):
    store = make_store(tmp_path)
    chunk = Chunk(
        id="doc-3",
        document_id="my-document",
        chunk_index=3,
        text="the backend uses FastAPI and Docker",
        embedding=[1.0, 0.0],
        metadata={"page": 5},
    )
    store.add([chunk])

    results = store.search_with_scores([1.0, 0.0])
    _, found = results[0]

    assert found.document_id == "my-document"
    assert found.chunk_index == 3
    assert found.text == "the backend uses FastAPI and Docker"
    assert found.metadata == {"page": 5}


def test_search_respects_top_k(tmp_path):
    store = make_store(tmp_path)
    store.add(
        [
            Chunk(id=f"doc-{i}", document_id="doc", chunk_index=i, text=str(i), embedding=[1.0, 0.0])
            for i in range(5)
        ]
    )

    results = store.search_with_scores([1.0, 0.0], top_k=2)

    assert len(results) == 2
