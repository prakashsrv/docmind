# DocMind

DocMind is a terminal-based AI assistant built from scratch, in phases, as a way to learn how a real Retrieval-Augmented Generation (RAG) system is put together — not by installing a framework, but by building each piece: an LLM client, a chat service, a PDF ingestion pipeline, and a semantic search engine.

This document explains both **what was built** and **why it works the way it does**, so it can be read as a project write-up or as a study guide for the concepts involved. For deeper dives, see [`docs/architecture.md`](docs/architecture.md) (component diagram, index-time vs. query-time flows) and [`docs/ingestion.md`](docs/ingestion.md) (the PDF → chunk → embedding pipeline in detail).

---

## What DocMind does today

DocMind is now a working Retrieval-Augmented Generation (RAG) system, built up in four phases:

1. A streaming, multi-turn terminal chatbot backed by Google's Gemini API.
2. A document-ingestion pipeline: PDF → text → overlapping chunks.
3. A semantic search layer: chunks → embeddings → an in-memory vector store you can query by meaning, not keywords.
4. A retrieval-augmented answer layer: question → relevant chunks → a grounded, cited answer from Gemini — instead of an answer from the model's training data alone.

Phases 1-3 and Phase 4 are still two separate entry points in the code (the interactive chatbot in `main.py` doesn't call into retrieval yet), but the full pipeline — PDF in, grounded answer with citations out — works end to end via `scripts/test_rag.py`. Everything below reflects what's actually implemented.

---

## Project structure

```
docmind/
├── app/
│   ├── core/
│   │   └── config.py            # All tunable constants in one place
│   ├── llm/
│   │   ├── client.py             # One shared Gemini client
│   │   ├── chat_service.py       # Multi-turn chat + streaming
│   │   └── prompts.py            # System prompt text
│   ├── models/
│   │   ├── document.py           # Document dataclass
│   │   └── chunk.py              # Chunk dataclass
│   ├── ingestion/
│   │   ├── pdf_loader.py         # PDF -> raw text
│   │   ├── chunker.py            # Text -> overlapping Chunk objects
│   │   └── pipeline.py           # Wires loader -> chunker -> embeddings
│   ├── embedding/
│   │   ├── embedding_service.py  # Text -> vector (Gemini embeddings)
│   │   └── vector_store.py       # In-memory store + cosine similarity search
│   ├── retrieval/
│   │   ├── retriever.py            # Question -> relevant Chunks (dense/embedding, similarity threshold applied)
│   │   ├── bm25_retriever.py       # Question -> relevant Chunks (keyword/BM25)
│   │   ├── hybrid_retriever.py     # Merges dense + BM25 via Reciprocal Rank Fusion
│   │   ├── reranker.py             # Cross-encoder pairwise (question, chunk) scoring
│   │   ├── reranking_retriever.py  # Wraps any retriever, reranks its candidates
│   │   └── rag_service.py          # Question -> grounded answer + sources
│   ├── storage/
│   │   └── chroma_store.py       # Persistent, disk-backed vector store (same interface as in-memory)
│   └── ui/
│       └── console.py            # Rich-based terminal rendering
├── scripts/
│   ├── test_pipeline.py          # Manual test: ingest + embed a PDF (in-memory, throwaway)
│   ├── test_search.py            # Manual test: ingest, then semantic search (in-memory)
│   ├── test_rag.py               # Manual test: full RAG loop, in-memory store
│   ├── test_hybrid.py            # Manual test: dense vs BM25 vs hybrid, side by side
│   ├── test_rerank.py            # Manual test: hybrid vs hybrid+cross-encoder-reranked
│   ├── ingest.py                 # Persist a PDF's embedded chunks into Chroma (run once per doc)
│   └── ask.py                    # Ask a question against the persisted Chroma store (no re-embedding)
├── tests/
│   ├── test_chunker.py               # Unit tests: chunking + overlap math
│   ├── test_embeddings.py            # Unit tests: EmbeddingService (Gemini API mocked)
│   ├── test_vector_store.py          # Unit tests: cosine similarity, ranking, top_k
│   ├── test_chroma_store.py          # Integration tests: ChromaVectorStore (real, on-disk, no mocks)
│   ├── test_retriever.py             # Unit tests: Retriever wiring + similarity threshold
│   ├── test_bm25_retriever.py        # Unit tests: keyword matching, zero-overlap filtering
│   ├── test_hybrid_retriever.py      # Unit tests: Reciprocal Rank Fusion, fallback to BM25
│   ├── test_reranker.py              # Unit tests: CrossEncoderReranker (model mocked)
│   └── test_reranking_retriever.py   # Unit tests: RerankingRetriever wiring (deps mocked)
├── docs/
│   ├── architecture.md           # Component diagram, index-time vs query-time flow
│   └── ingestion.md              # PDF -> chunk -> embedding pipeline, in depth
├── screenshots/                  # Terminal screenshots for reviewers (see screenshots/README.md)
├── data/
│   ├── documents/                # Source PDFs (gitignored, except .gitkeep)
│   └── chroma/                   # Persisted Chroma database (gitignored, regenerate via scripts/ingest.py)
├── conftest.py                   # Puts the project root on sys.path for pytest
├── main.py                       # Entry point for the interactive chatbot
└── .env                          # GEMINI_API_KEY (gitignored)
```

Each folder has exactly one job. This mirrors a pattern you'd recognize from mobile or backend architecture: a client/networking layer, a service/business-logic layer, a data/model layer, and a UI layer, each of which can change independently of the others.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install google-genai python-dotenv rich pymupdf chromadb rank-bm25
pip install sentence-transformers   # optional, only needed for reranking (see below) -- pulls in torch, a large download
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Never commit `.env` — it's already in `.gitignore`.

---

## Running it

**Chat with Gemini:**

```bash
python main.py
```

**Ingest a PDF and inspect its chunks/embeddings:**

```bash
python scripts/test_pipeline.py data/documents/your_file.pdf
```

**Run a semantic search over a PDF:**

```bash
python scripts/test_search.py data/documents/your_file.pdf "your query here"
```

**Ask a grounded, cited question over a PDF (full RAG, in-memory):**

```bash
python scripts/test_rag.py data/documents/your_file.pdf "your question here"
```

**Persist a PDF to disk, then ask questions without re-embedding it:**

```bash
python scripts/ingest.py data/documents/your_file.pdf   # run once per document
python scripts/ask.py "your question here"               # run any time after
```

**Compare dense-only vs. keyword-only vs. hybrid retrieval for a question:**

```bash
python scripts/test_hybrid.py data/documents/your_file.pdf "your question here"
```

**Compare hybrid retrieval before and after cross-encoder reranking (requires `sentence-transformers`):**

```bash
python scripts/test_rerank.py data/documents/your_file.pdf "your question here"
```

---

## Tests

Unit tests cover the pieces that don't need a live API call to verify — chunking math and `Retriever`/`EmbeddingService` wiring, with Gemini calls mocked out via `unittest.mock`. They run offline and don't touch your `.env` or API quota.

```bash
pip install pytest
pytest
```

The `scripts/test_*.py` files are a different thing: manual, end-to-end smoke tests that do call the real Gemini API against a real PDF, meant to be run by hand while developing rather than as part of an automated test suite.

---

## Part 1 — The chatbot (`app/llm/`)

### A single, shared client

`client.py` builds one `genai.Client` at import time using the API key from `.env`, and every other module imports that same instance. This avoids re-authenticating per request and keeps the one piece of code that touches credentials isolated from everything else.

```python
load_dotenv(override=True)
client = genai.Client(api_key=api_key)
```

`override=True` matters more than it looks: by default, `load_dotenv()` refuses to overwrite a variable that's already set in the shell's environment. Without it, a stale exported `GEMINI_API_KEY` from an earlier terminal session can silently shadow the real key in `.env`, and you get "API key not valid" errors even with a correct `.env` file.

### Why streaming instead of a single response

An LLM API call has two shapes. `generate_content` blocks until the entire answer has been generated, then returns it as one string — simple, but you wait several seconds staring at nothing. `generate_content_stream` (used here via `send_message_stream`) instead returns an **iterator**: the server sends the response in pieces as it's generated, and you can start displaying it immediately.

```python
def stream(self, message: str):
    response = self.chat.send_message_stream(message)
    for chunk in response:
        if chunk.text:
            yield chunk.text
```

Making this a generator (`yield`, not `return`) is the important part — it means `ChatService` hands each fragment to the caller the moment it arrives, rather than collecting the whole response into a list first and returning it at the end. The UI layer (`console.py`) then redraws a live-updating panel every time a new fragment comes in, which is what produces the "typing" effect.

### Multi-turn conversation and why it's not automatic

By default, every call to an LLM API is **stateless** — the model has no memory of what you said a moment ago unless you resend it. "Chat history" is really just: keep a running list of every user message and every model reply, and resend that whole list with each new request.

Rather than hand-maintaining that list, `ChatService` uses `client.chats.create(...)`, which returns a session object that does this bookkeeping internally. Every `send_message()` call appends to its own history and automatically includes prior turns in the next request, which is why you can ask "explain RAG" and then "give an example" and get a coherent follow-up instead of a non-sequitur.

### Resilience: error handling and logging

Both `ask()` and `stream()` wrap their network calls in `try/except`, log the failure, and re-raise rather than swallowing it. `main.py` is the layer that actually decides what to do about a failure — it catches the exception around the call site and prints a friendly message instead of crashing:

```python
try:
    stream_response(service.stream(question))
except Exception as e:
    print_error(str(e))
```

Logging (`logging.getLogger("docmind")`) replaces scattered `print()` debugging with leveled, timestamped output (`INFO`, `ERROR`) that can be filtered or redirected without touching call sites — standard practice once a project grows past a toy script.

### Configuration over hardcoding

`app/core/config.py` holds every tunable value as a named constant — model name, temperature, chunk size, embedding model, and so on. Nothing in the codebase hardcodes `"gemini-2.5-flash"` inline; it all reads from `config.MODEL_NAME`. Change the model or a parameter in one place, and everything downstream picks it up.

---

## Part 2 — Turning documents into searchable knowledge (`app/ingestion/`, `app/embedding/`)

This is the part that makes DocMind a *document* assistant rather than just a chatbot, and it's the foundation of RAG. The theory first, then how the code maps onto it.

### The problem: LLMs can't read a 500-page PDF

You can't just paste an entire document into a prompt — it may not fit in the model's context window, and even when it does, stuffing in irrelevant pages wastes tokens and dilutes the model's attention on what actually matters. So instead of "read the whole document," the strategy is: **find the small number of passages that are actually relevant to the question, and only send those.**

That requires three things: breaking documents into small, retrievable pieces (chunking), representing each piece in a form you can search by *meaning* rather than exact keywords (embeddings), and a way to store and query those representations (a vector store). DocMind implements all three.

### Step 1 — Extraction (`pdf_loader.py`)

`load_pdf_text()` uses PyMuPDF (`fitz`) to open a PDF and pull the text layer off every page, concatenating them into one string. It deliberately knows nothing about chunking or embeddings — its only job is *bytes on disk → plain text*. Real-world PDFs are messy (multi-column layouts, headers/footers, tables), so this step is often the least reliable part of a RAG pipeline in practice, even though it looks the simplest.

### Step 2 — The data model (`app/models/`)

Two small dataclasses hold everything else together:

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

A `Chunk` remembers which `Document` it came from (`document_id`) and its position within it (`chunk_index`), so a search result can always be traced back to its source. The `embedding` field starts as `None` and gets filled in once the chunk has been through the `EmbeddingService` — the chunk *owns* its own vector rather than keeping vectors in a separate parallel list.

### Step 3 — Chunking (`chunker.py`)

Splitting text isn't just "cut every 500 characters." `chunk_document()` uses a **sliding window with overlap**:

```
chunk_size = 500, overlap = 100  →  step = 400

[  chunk 1: chars 0–500   ]
        [  chunk 2: chars 400–900   ]
                [  chunk 3: chars 800–1300  ]
```

Each chunk starts 400 characters after the previous one, so the last 100 characters of one chunk reappear at the start of the next. Without overlap, a sentence or idea that happens to fall right on a chunk boundary gets split in half, and neither half alone captures its meaning — which then hurts retrieval later, since that idea may not surface as a strong match to any query. Overlap trades a small amount of redundancy for a much lower chance of losing context at the seams.

Chunk size itself is a tuning knob with a real trade-off: smaller chunks are more precise (a match is very specifically about the query) but lose surrounding context; larger chunks preserve more context but dilute precision and cost more tokens once they're sent to the LLM. `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP` exist so this can be tuned in one place.

### Step 4 — Embeddings (`embedding_service.py`)

An embedding model converts a piece of text into a vector — a list of a few hundred to a few thousand floating-point numbers — such that pieces of text with *similar meaning* end up as vectors that point in *similar directions* in that space. This is what makes semantic search possible: "mobile app experience" and "developed Android applications" can match even though they don't share a single keyword, because the model places them near each other in vector space.

```python
def embed(self, text: str) -> list[float]:
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL_NAME,  # "gemini-embedding-001"
        contents=text,
    )
    return response.embeddings[0].values
```

`embed_chunks()` embeds every chunk of a document in a **single batched API call** rather than looping and calling the API once per chunk — fewer network round trips for the same result, which matters once you're ingesting documents with dozens or hundreds of chunks.

### Step 5 — The vector store (`vector_store.py`)

This is deliberately the simplest possible implementation, not Chroma or Pinecone or any production vector database — the goal was to understand what those tools actually do internally before treating them as a black box.

`InMemoryVectorStore` is just a Python list of chunks. `add()` appends to it. `search()` compares a query vector against every stored chunk's embedding using **cosine similarity** and returns the top-k closest matches:

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
```

Cosine similarity measures the *angle* between two vectors rather than their distance or magnitude — a score of 1.0 means they point in exactly the same direction (same meaning), 0 means unrelated, -1 means opposite. Dividing by both vectors' lengths (norms) is what cancels out magnitude, so two vectors that point the same way score highly whether the text was short or long.

This implementation does a linear scan — it checks every stored chunk on every search, which is `O(n)`. That's fine for hundreds of chunks and exactly the part that a real vector database optimizes away with indexing (e.g. HNSW graphs), while keeping the same conceptual `add()` / `search()` interface. Swapping in Chroma later is meant to be closer to a one-line change than a redesign, because the interface was designed first.

### Step 6 — The full pipeline (`pipeline.py`)

```
PDF → load_pdf_text() → Document → chunk_document() → Chunks
    → EmbeddingService.embed_chunks() → Chunks (with embeddings)
```

`process_pdf(path)` runs all of this in one call and returns `(document, chunks)`, where every chunk already has its `.embedding` populated — ready to be added to a vector store and searched.

---

## Part 3 — Retrieval-Augmented Generation (`app/retrieval/`)

This is the phase that turns "a chatbot" plus "a search demo" into an actual RAG system: the model's answer is now grounded in retrieved evidence instead of only its training data.

```
User Question
      │
      ▼
   Retriever            (embed question, search the vector store)
      │
 Top-k Chunks
      │
      ▼
 Prompt Builder          (build_rag_prompt: label chunks, add instructions)
      │
      ▼
  Gemini LLM             (complete: stateless, one-shot)
      │
      ▼
Grounded Answer + Sources
```

### The Retriever never talks to Gemini

`Retriever.retrieve(question, top_k)` does exactly two things: embed the question with `EmbeddingService`, then search an `InMemoryVectorStore` for the closest chunks. It has no idea an LLM exists. This separation means the retrieval step — the part most likely to get swapped out later for hybrid search or reranking — can change without touching how answers are generated, and can be tested/reasoned about in isolation.

### Building a prompt that forces grounding

Instead of sending the question straight to Gemini, `build_rag_prompt()` (in `app/llm/prompts.py`) wraps it with the retrieved chunks and explicit instructions:

```python
RAG_PROMPT_TEMPLATE = """You are DocMind.

Answer the question using ONLY the context below. Do not use outside
knowledge, and do not invent facts that aren't in the context.

If the answer is not present in the context, respond with exactly this
sentence and nothing else:
"{not_found}"

Context:
{context}
...
```

Each chunk is labeled `[Chunk N]` in the context block, which is what lets both the model and the code refer back to *which* piece of evidence an answer came from.

### `RagService`: stateless by design

`RagService.answer(question)` chains `Retriever` → `build_rag_prompt` → a new `complete(prompt)` function in `chat_service.py`. `complete()` is deliberately a **one-shot, stateless** call (`client.models.generate_content`), not the multi-turn `chat.send_message()` used by the interactive chatbot — a RAG prompt already carries its full context every time, so it shouldn't also accumulate into a growing conversation history the way a normal chat turn does.

### Citations that can't be hallucinated

The tempting approach is to just ask the model to cite its sources in the answer text and trust it. Instead, `RagService` attaches sources **deterministically**, from the chunks it actually retrieved and sent to Gemini — not from what the model claims it used:

```python
sources = ", ".join(f"Chunk {chunk.chunk_index}" for chunk in chunks)
return f"{answer}\n\nSources: {sources}"
```

The model can still get the *answer* wrong, but it can't fabricate a citation to a chunk that was never in its prompt — the set of possible citations is constrained by code, not by the model's honesty.

### The "I don't know" fallback

`NOT_FOUND_MESSAGE` is a single constant shared between the prompt (which tells Gemini to return it verbatim when the answer isn't in the context) and `RagService` (which checks for it to decide whether attaching "Sources:" even makes sense). This used to rely entirely on the model following instructions, since vector search always returned *something*. That gap is now closed by the similarity threshold below — `Retriever` can now return zero chunks, which triggers this fallback deterministically rather than hoping the model behaves.

---

## Part 4 — Retrieval quality: a similarity threshold (`app/retrieval/retriever.py`)

Version 1 had a real problem, easiest to see with an example: imagine a knowledge base containing a resume, an Android guide, a Flutter cookbook, and a Python guide. Ask "What is Jetpack Compose?" and dense vector search might rank the Flutter guide and the resume above the actually-relevant Android guide — embeddings capture meaning well, but not perfectly, and "top-k" search has no concept of "good enough," only "best available." Ask something with *no* relevant answer anywhere in the store, and the same problem gets worse: search still confidently returns its top-k least-bad matches, and the LLM gets fed irrelevant context that invites a hallucinated answer.

The fix: `Retriever.retrieve()` now checks each candidate's cosine similarity score against `config.SIMILARITY_THRESHOLD` (default `0.70`) and drops anything below it, rather than trusting `top_k` alone:

```python
scored = self.vector_store.search_with_scores(query_vector, top_k=top_k)
accepted = [(score, chunk) for score, chunk in scored if score >= min_score]
return [chunk for _, chunk in accepted]
```

If nothing clears the bar, `retrieve()` returns `[]`, `RagService` sees an empty chunk list, and returns `NOT_FOUND_MESSAGE` immediately — no LLM call needed, and no dependence on the model choosing to admit uncertainty.

This required exposing scores, not just chunks, out of the vector store: `InMemoryVectorStore.search_with_scores()` returns `list[tuple[float, Chunk]]` instead of just `list[Chunk]`, sharing its ranking logic with the original `search()` via a private `_ranked()` helper so there's only one place that actually computes similarity and sorts.

**On the threshold value itself:** `0.70` is a starting guess, explicitly not a calibrated one — cosine similarity distributions depend on the embedding model, and setting this too high silently rejects real answers as "not found," while too low defeats the purpose. `Retriever` logs the best score and rejection count on every call, and `scripts/test_search.py` now prints per-chunk scores specifically so this number can be tuned against real queries and documents instead of guessed. Treat it as a knob to revisit once you have more usage data — this is exactly the kind of thing `docs/architecture.md`'s "Retrieval quality" section and the Week 3 evaluation harness (below) are meant to make measurable rather than eyeballed.

## Persistence: swapping in ChromaDB (`app/storage/chroma_store.py`)

Every script up to this point rebuilt its vector store from scratch on every run: `process_pdf()` re-extracts, re-chunks, and re-embeds a PDF every single time, which is fine for experimentation but means a real application would redo (and re-pay for) that work on every restart. `ChromaVectorStore` fixes that by writing embeddings to disk instead of a Python list that dies when the process exits.

The interface is identical to `InMemoryVectorStore` on purpose — `add(chunks)`, `search(query_vector, top_k)`, `search_with_scores(query_vector, top_k)` — because `Retriever` only ever calls those three methods and has no idea which implementation it's holding:

```python
Retriever(InMemoryVectorStore())   # rebuilt every run
Retriever(ChromaVectorStore())      # persisted on disk, instant on restart
```

A few things worth understanding about how the swap actually works, not just that it works:

- **Cosine distance vs. cosine similarity.** Chroma's default distance metric is squared L2; the collection is created with `metadata={"hnsw:space": "cosine"}` so it computes cosine distance instead, which is `1 - cosine_similarity`. `ChromaVectorStore` converts that back (`similarity = 1 - distance`) so `SIMILARITY_THRESHOLD` and the rest of the app never need to know Chroma uses a different convention than `vector_store.py`'s own `cosine_similarity()`.
- **Deterministic IDs enable upserting.** Chroma's `add()` errors on duplicate ids, and a naive `add()`-every-time would either crash or (if you switch to blind upserts) silently duplicate every chunk on every re-run. `chunk.id` is now derived as `f"{document.id}-{chunk_index}"`, and `document.id` is a hash of the extracted text (`app/ingestion/pipeline.py`) rather than a random `uuid4`. Re-ingesting the same PDF now produces the exact same ids every time, so `collection.upsert()` overwrites the existing rows in place instead of accumulating duplicates.
- **Metadata has to be flat.** Chroma's metadata values must be strings/ints/floats/bools, not nested structures, so `Chunk.metadata` (an arbitrary dict) is JSON-serialized into a single `metadata_json` string field and parsed back out on read, rather than passed through directly.
- **A real behavioral difference from `InMemoryVectorStore`:** chunks coming back from `ChromaVectorStore.search()` have `embedding=None` — the vector isn't requested back from Chroma on query, since nothing downstream (`RagService`, citations) needs it after retrieval. `InMemoryVectorStore` happens to return the original `Chunk` object with its embedding intact, purely as a side effect of being backed by a plain list. Don't rely on `chunk.embedding` being populated after a search with either store.

New scripts reflect the resulting two-step workflow: `scripts/ingest.py <pdf>` does the expensive part once (extract, chunk, embed, persist), and `scripts/ask.py "<question>"` does the cheap part (open the existing Chroma collection, embed only the question, search, generate) — which is the "Load Chroma → Ready" startup the mentor plan describes, versus `scripts/test_rag.py`'s "reprocess everything, every time."

`tests/test_chroma_store.py` covers this with real Chroma instances against a throwaway `tmp_path` directory per test (not mocks — Chroma is a local embedded database with no network calls, so there's no reason to fake it) — upserting, empty-store behavior, metadata round-tripping, and `top_k`.

---

## Part 5 — Hybrid retrieval: BM25 + dense search (`app/retrieval/bm25_retriever.py`, `hybrid_retriever.py`)

Dense (embedding) search understands meaning but not always precise wording. Ask about "PyMuPDF" and an embedding model represents the *overall topic* of a chunk (PDF processing, say) rather than weighting an exact library name heavily — so a chunk that literally contains "PyMuPDF" might not be the top cosine-similarity match. BM25, classic keyword/term-frequency search, has the opposite strength and weakness: it finds exact terms reliably but has no idea that "car" and "automobile" mean the same thing. Combining both catches what either one misses alone.

### `BM25Retriever`: the keyword half

`BM25Retriever` (using the `rank-bm25` package's `BM25Okapi`) takes the *entire* chunk corpus up front, unlike `Retriever`, which searches a vector store per query — BM25 needs to see every document to compute how rare/informative each term is across the corpus (its "inverse document frequency"):

```python
def __init__(self, chunks: list[Chunk]):
    self.chunks = chunks
    self._index = BM25Okapi([_tokenize(chunk.text) for chunk in chunks]) if chunks else None
```

A subtlety worth calling out: `retrieve()` only returns chunks with a BM25 score strictly greater than zero. A score of exactly 0 means no term overlap at all — verified directly (`bm25.get_scores(...)` returns `0.0` for a chunk sharing no words with the query) — so this is BM25's own signal that a chunk isn't a match, not a weak one, and it's the keyword-search equivalent of `Retriever`'s similarity threshold: don't return your "best" match if it isn't actually related to the question.

### Fusing two different scoring scales: Reciprocal Rank Fusion

Dense scores (cosine similarity) live on a 0-1 scale; BM25 scores are unbounded and depend on the size and content of the corpus. A BM25 score of 4.2 says nothing about whether it's better or worse than a cosine similarity of 0.85 — they can't be averaged or compared directly. `hybrid_retriever.py` sidesteps this with **Reciprocal Rank Fusion (RRF)**, which ignores raw scores entirely and fuses based on *rank position* in each list instead:

```python
fused_score(chunk) = sum over each list it appears in of 1 / (k + rank)
```

A chunk ranked #1 by both dense and BM25 search clearly deserves to outrank one ranked #1 by only one of them — and that reasoning holds regardless of what scale either method's underlying scores are on. `k=60` is the standard smoothing constant from the original RRF paper, not something tuned for this project specifically.

**Worked example.** Say dense search and BM25 each rank the same 4 chunks differently:

| Chunk | Dense rank | BM25 rank | Fused score = 1/(60+dense) + 1/(60+bm25) |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.0323 |
| B | 2 | 1 | 1/62 + 1/61 = 0.0325 |
| C | 4 | 2 | 1/64 + 1/62 = 0.0318 |
| D | 5 | 10 | 1/65 + 1/70 = 0.0297 |

Final fused order: **B → A → C → D.** B edges out A despite ranking one position lower in dense search, because it ranks higher in BM25 (#1 vs #3) — RRF rewards a chunk that both methods agree is good over one only a single method loves, even if that one method ranks it #1. (If you've seen this example elsewhere with the order listed as A → B → C → D, double-check the arithmetic — with `k=60` these exact rank pairs put B slightly ahead of A.)

### `HybridRetriever`: each side contributes only what it's confident about

```python
dense_candidates = self.dense_retriever.retrieve(question, top_k=candidate_pool)   # threshold already applied
bm25_candidates = [chunk for _, chunk in self.bm25_retriever.retrieve(question, top_k=candidate_pool)]  # zero-overlap already filtered

if not dense_candidates and not bm25_candidates:
    return []

return _reciprocal_rank_fusion([dense_candidates, bm25_candidates])[:top_k]
```

This is deliberately built on top of `Retriever.retrieve()` and `BM25Retriever.retrieve()` as-is, not their raw unfiltered scores — each one already decides what it's confident about (the similarity threshold on one side, the zero-overlap filter on the other) before hybrid fusion ever sees the candidates. That preserves the "I don't know" guarantee from Part 4: if dense search found nothing above threshold *and* BM25 found no keyword overlap at all, `HybridRetriever` returns `[]`, same as a plain `Retriever` would. But if dense search comes back empty while BM25 finds an exact keyword match, hybrid retrieval surfaces it anyway — which is the actual point of building this in the first place, demonstrated directly in `tests/test_hybrid_retriever.py::test_hybrid_retriever_falls_back_to_bm25_when_dense_finds_nothing`.

`HybridRetriever` exposes the same `retrieve(question, top_k) -> list[Chunk]` signature as `Retriever`, so it's a drop-in replacement anywhere a `Retriever` is used, including `RagService` — `RagService(HybridRetriever(dense, bm25))` instead of `RagService(Retriever(store))`, no changes needed to `RagService` itself.

`scripts/test_hybrid.py` runs dense-only, BM25-only, and fused hybrid retrieval against the same question side by side, specifically so the difference is visible rather than theoretical.

---

## Part 6 — Cross-encoder reranking (`app/retrieval/reranker.py`, `reranking_retriever.py`)

Hybrid retrieval (Part 5) improves *which* chunks make it into the candidate set, but nothing about dense search, BM25, or RRF actually reads a question and a chunk together and judges "does this specific pairing answer that specific question." A cross-encoder does exactly that:

```
Top ~20 candidates (from hybrid retrieval)
                │
                ▼
   Cross-encoder scores each (question, chunk) pair directly
                │
                ▼
        Best 5, re-ordered by that score
                │
                ▼
               LLM
```

### Why this is a different kind of model, not just a better one

Dense search and BM25 both work by computing a representation of the question and a representation of each chunk *independently*, then comparing those two fixed representations (cosine similarity for embeddings; term-frequency statistics for BM25). That independence is exactly what makes them fast — a chunk's embedding or BM25 statistics can be computed once and reused for every future query, which is why they can scale to searching an entire corpus.

A cross-encoder gives up that independence on purpose: it feeds the question and a single candidate chunk into the model *together*, in one forward pass, and the model directly predicts how relevant that specific pairing is — closer to actually reading both texts side by side than comparing two points in space. That tends to be meaningfully more accurate, but there's no way to precompute anything, since the score only exists for a specific (question, chunk) pair. That's precisely why it's used to re-score a modest ~20-candidate shortlist from hybrid retrieval rather than the whole corpus — precision on a shortlist, not search over everything.

### The code

`app/retrieval/reranker.py`'s `CrossEncoderReranker.rerank(question, chunks, top_k)` scores every `(question, chunk.text)` pair with a `sentence-transformers` `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default — a small, fast model trained specifically for passage reranking) and returns the top-k chunks by that score, discarding the base retriever's original order entirely.

`app/retrieval/reranking_retriever.py`'s `RerankingRetriever` wraps *any* retriever — a plain `Retriever`, a `HybridRetriever`, whatever exposes `retrieve(question, top_k)` — pulls a wider candidate pool (`config.RERANK_CANDIDATE_POOL`, default 20) from it, and hands those candidates to the reranker:

```python
RagService(RerankingRetriever(HybridRetriever(dense, bm25)))
```

Crucially, reranking doesn't add its own relevance gate — it only re-orders whatever the base retriever already decided was plausible. If the base retriever (dense threshold, BM25 zero-overlap filter, or their hybrid combination) finds nothing, `RerankingRetriever` returns `[]` without ever touching the cross-encoder, so the "I don't know" guarantee from Parts 4-5 still holds all the way through reranking.

### A dependency worth knowing about: lazy-loaded on purpose

`sentence-transformers` pulls in `torch`, by far the heaviest dependency in this project (hundreds of MB). Importing `app.retrieval.reranker` does **not** require `torch` or `sentence-transformers` to be installed — the `from sentence_transformers import CrossEncoder` import is deferred inside `_get_model()`, which only runs the first time `.rerank()` is actually called, and the loaded model is cached at module level afterward so it only loads once per process, not once per call. This is why `tests/test_reranker.py` and `tests/test_reranking_retriever.py` can run (and did, as part of the 43-test suite) without the real dependency installed at all — they mock `_get_model()` directly. Only `scripts/test_rerank.py`, which needs the real model, requires `pip install sentence-transformers` first.

`scripts/test_rerank.py` runs hybrid retrieval both before and after reranking for the same question, so the reordering is visible rather than theoretical — this comparison (hybrid alone vs. hybrid + reranked) is exactly what the next phase's evaluation harness is meant to turn into an actual number.

---

## What's next

Per the guide this project follows, DocMind (Project 1) is now through Stage 5 of 7: LLM API foundations, ingestion/chunking, embeddings + vector DB, hybrid retrieval + reranking, and grounded generation with citations are all built. What's left:

**Stage 6 — the evaluation harness.** Explicitly "the hero feature" of the whole project: a golden dataset of roughly 50 question/answer pairs against your own documents, and a script that runs the full pipeline against every question and measures two things — retrieval quality (did it fetch the right chunks?) and faithfulness (did the answer actually stick to the retrieved context, or hallucinate?). The entire point of building the eval harness now, rather than earlier, is to run it against the retrieval pipeline in two configurations — hybrid alone, and hybrid + reranked — and produce the headline number: "retrieval relevance went from X to Y after adding reranking." That's the interview-ready line this whole project has been building toward, and it only means something because both configurations already exist to compare (Parts 5 and 6, above).

**Stage 7 — serving it.** A FastAPI service (`/upload`, `/query` endpoints), Docker packaging, and per-query cost/latency logging (tokens in/out, dollars per query) — turning DocMind from a library plus test scripts into an actual running service, which is also the shape `TaskAgent` (a separate, tool-using-agent project) expects to call as one of its tools.

Smaller open items along the way: richer chunk metadata (page number, section) so citations can eventually say "Page 5" instead of "Chunk 3"; wiring `RagService`/the retrieval stack into the interactive `main.py` chat loop so it's one application instead of a library plus test scripts; applying the existing `SYSTEM_PROMPT` to the chat session; and optionally swapping `ChromaVectorStore` for `pgvector` later, to learn a production-grade Postgres-backed store once Chroma's local-first approach has served its purpose.

---

## Concepts this project demonstrates

For anyone skimming this as a portfolio piece or before an interview, the ideas exercised here are: streaming API responses via generators/iterators, stateless-vs-stateful API design (why chat history has to be managed explicitly, and why a RAG call intentionally opts back out of it), the extract → chunk → embed → index → retrieve pipeline that underlies every production RAG system, cosine similarity and vector search implemented from first principles, prompt engineering to force grounding and a verbatim refusal message, deterministic citation attachment as a way to avoid trusting model self-reporting, a similarity threshold as the mechanism that lets a retriever say "I don't know" instead of always returning its best-available guess, designing an interface (`add`/`search`/`search_with_scores`) so a storage backend can be swapped from an in-memory list to a persistent database without touching any calling code, deterministic ID generation as a prerequisite for idempotent upserts, dense vs. keyword (BM25) search as complementary rather than competing signals, Reciprocal Rank Fusion as a way to merge rankings from incomparable scoring scales without normalizing either one, bi-encoder vs. cross-encoder architectures as a precompute-vs-precision trade-off, lazy imports as a way to keep a heavy optional dependency from leaking into every other module's testability, and a layered architecture (client / service / model / retrieval / storage / UI) that keeps each concern independently testable and replaceable.
