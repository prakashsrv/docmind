import pytest

from app.ingestion.chunker import _split_text, chunk_document
from app.models.document import Document


def make_document(text: str) -> Document:
    return Document(id="doc-1", name="test.txt", text=text)


def test_split_text_produces_overlapping_windows():
    text = "a" * 1000
    parts = _split_text(text, chunk_size=500, overlap=100)

    assert len(parts) == 3
    assert parts[0] == text[0:500]
    assert parts[1] == text[400:900]
    assert parts[2] == text[800:1000]


def test_chunk_document_wraps_parts_in_chunk_objects():
    document = make_document("x" * 1000)
    chunks = chunk_document(document, chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].embedding is None


def test_chunk_document_rejects_overlap_greater_than_or_equal_to_chunk_size():
    document = make_document("short text")

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=100, overlap=200)


def test_short_text_produces_a_single_chunk():
    document = make_document("hello world")
    chunks = chunk_document(document, chunk_size=500, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].chunk_index == 0
