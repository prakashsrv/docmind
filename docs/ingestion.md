# Ingestion

How a PDF becomes a set of embedded, searchable `Chunk` objects. This covers the same ground as `README.md`'s "Part 2" section but focused purely on this one pipeline, for anyone who wants to read or modify just the ingestion path.

## Pipeline

```
PDF file
   │
   ▼
load_pdf_text(path)        app/ingestion/pdf_loader.py
   │
   ▼
Document(id, name, text)   app/models/document.py
   │
   ▼
chunk_document(document)   app/ingestion/chunker.py
   │
   ▼
[Chunk, Chunk, ...]        app/models/chunk.py  (embedding=None)
   │
   ▼
EmbeddingService.embed_chunks(chunks)   app/embedding/embedding_service.py
   │
   ▼
[Chunk, Chunk, ...]        (embedding populated)
```

All of the above is `process_pdf(path)` in `app/ingestion/pipeline.py`, which returns `(document, chunks)`.

## Extraction (`pdf_loader.py`)

Uses PyMuPDF (`fitz`) to open the PDF and call `.get_text()` on every page, joining the results with newlines. This step is intentionally "dumb" — it has no idea chunking or embeddings exist. In practice, extraction quality is the least predictable part of any RAG pipeline: multi-column layouts, tables, and headers/footers can all come out of `get_text()` in a different order than a human would read them. If retrieval quality is ever surprisingly bad, check the raw extracted text first before assuming the problem is in chunking or embeddings.

## Chunking (`chunker.py`)

A **sliding window with overlap**, not a naive fixed-size split:

```
chunk_size = 500, overlap = 100  →  step = chunk_size - overlap = 400

[  chunk 0: chars 0–500    ]
        [  chunk 1: chars 400–900   ]
                [  chunk 2: chars 800–1300  ]
```

Why overlap exists: without it, an idea or sentence that happens to land on a chunk boundary gets cut in half, and neither half alone represents it well enough to match a relevant query later. Overlap means the last `overlap` characters of one chunk are repeated at the start of the next, so that content still shows up whole in at least one chunk.

Why chunk size is a trade-off, not just a constant: smaller chunks are more precise (a match is specifically about the query) but can lose the surrounding context a full answer needs; larger chunks preserve context but dilute precision and cost more tokens once sent to the LLM. `config.CHUNK_SIZE` and `config.CHUNK_OVERLAP` (`app/core/config.py`) are the two knobs — tune them per document type rather than assuming the defaults are universally right.

Each resulting `Chunk` carries `document_id` (which `Document` it came from) and `chunk_index` (its position) — both are needed later to trace a retrieved result back to where it came from, and to build citations.

## Data model (`app/models/`)

```python
@dataclass
class Document:
    id: str
    name: str
    text: str

@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)
```

A chunk owns its own embedding rather than keeping vectors in a separate parallel array — this is what makes `Chunk` the single unit that flows through chunking, embedding, storage, and retrieval without ever needing to be re-associated with the right vector by index.

## Testing this pipeline

- `scripts/test_pipeline.py <pdf>` — runs `process_pdf()` and prints character count, chunk count, and the first few dimensions of chunk 0's embedding.
- `tests/test_chunker.py` — unit tests for `chunk_document()` and its internal `_split_text()` helper: correct window/overlap math, the single-chunk edge case for short text, and the `overlap >= chunk_size` validation error. These run without any network access or API key.
- `tests/test_embeddings.py` — unit tests for `EmbeddingService`, with `client.models.embed_content` mocked out, verifying that `embed_chunks()` makes exactly one batched API call (not one per chunk) and assigns each result to the right chunk.
