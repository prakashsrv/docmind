from app.embedding.vector_store import InMemoryVectorStore, cosine_similarity
from app.models.chunk import Chunk


def test_cosine_similarity_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_handles_zero_vector_without_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_search_with_scores_ranks_best_match_first():
    store = InMemoryVectorStore()
    close_match = Chunk(id="1", document_id="d", chunk_index=0, text="close", embedding=[1.0, 0.0])
    far_match = Chunk(id="2", document_id="d", chunk_index=1, text="far", embedding=[0.0, 1.0])
    store.add([far_match, close_match])

    results = store.search_with_scores([1.0, 0.0], top_k=2)

    assert [chunk.id for _, chunk in results] == ["1", "2"]
    assert results[0][0] == 1.0
    assert results[1][0] == 0.0


def test_search_with_scores_respects_top_k():
    store = InMemoryVectorStore()
    store.add(
        [
            Chunk(id=str(i), document_id="d", chunk_index=i, text=str(i), embedding=[1.0, 0.0])
            for i in range(5)
        ]
    )

    results = store.search_with_scores([1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_ignores_chunks_without_an_embedding():
    store = InMemoryVectorStore()
    store.add([Chunk(id="1", document_id="d", chunk_index=0, text="no vector yet")])

    results = store.search_with_scores([1.0, 0.0])

    assert results == []
