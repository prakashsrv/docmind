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
│   │   ├── retriever.py          # Question -> relevant Chunks
│   │   └── rag_service.py        # Question -> grounded answer + sources
│   └── ui/
│       └── console.py            # Rich-based terminal rendering
├── scripts/
│   ├── test_pipeline.py          # Manual test: ingest + embed a PDF
│   ├── test_search.py            # Manual test: ingest, then semantic search
│   └── test_rag.py               # Manual test: full RAG loop, question -> cited answer
├── tests/
│   ├── test_chunker.py           # Unit tests: chunking + overlap math
│   ├── test_embeddings.py        # Unit tests: EmbeddingService (Gemini API mocked)
│   └── test_retriever.py         # Unit tests: Retriever wiring (deps mocked)
├── docs/
│   ├── architecture.md           # Component diagram, index-time vs query-time flow
│   └── ingestion.md              # PDF -> chunk -> embedding pipeline, in depth
├── screenshots/                  # Terminal screenshots for reviewers (see screenshots/README.md)
├── data/
│   └── documents/                # Source PDFs (gitignored, except .gitkeep)
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
pip install google-genai python-dotenv rich pymupdf
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

**Ask a grounded, cited question over a PDF (full RAG):**

```bash
python scripts/test_rag.py data/documents/your_file.pdf "your question here"
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

### The "I don't know" fallback, and its current limitation

`NOT_FOUND_MESSAGE` is a single constant shared between the prompt (which tells Gemini to return it verbatim when the answer isn't in the context) and `RagService` (which checks for it to decide whether attaching "Sources:" even makes sense). Worth being honest about a real gap here: cosine similarity search always returns *something* — the top-k least-bad matches — even for a question with no relevant content in the document at all, like "who won the World Cup" against a document about RAG architecture. There's no similarity-score threshold filtering out weak matches before they reach the prompt, so today the fallback relies entirely on the model following instructions rather than on retrieval refusing to return irrelevant chunks. That's precisely the gap that reranking and score thresholds (below) are meant to close.

---

## What's next

Per the roadmap this project is following, Version 1 (chat, ingestion, chunking, embeddings, semantic search, retrieval, RAG service, citations, "I don't know" fallback) is functionally complete. The remaining work is about **improving retrieval quality**, not making the system work in the first place:

```
Current                              Planned

Dense (embedding) search             BM25 keyword search
      │                                    +
      ▼                              Dense (embedding) search
    LLM                                    │
                                            ▼
                                        Merge results
                                            │
                                            ▼
                                   Cross-Encoder Reranker
                                            │
                                            ▼
                                           LLM
```

This is **hybrid retrieval**: dense (embedding) search is good at matching meaning but can miss exact keywords/names/codes that BM25 (classic keyword search) catches reliably; combining both and reranking the merged results is one of the highest-leverage improvements to retrieval quality in a real RAG system.

Concretely, still open: replacing `InMemoryVectorStore` with a persistent vector database (Chroma), adding BM25 and merging it with dense search, a cross-encoder reranker, a similarity-score threshold so retrieval can refuse to hand over irrelevant chunks, an evaluation harness (RAGAS/DeepEval) to measure retrieval and answer quality objectively rather than by eyeballing outputs, wiring `RagService` into the interactive `main.py` chat loop so it's one application instead of a library plus test scripts, applying the existing `SYSTEM_PROMPT` to the chat session, and eventually a FastAPI service + Docker packaging.

---

## Concepts this project demonstrates

For anyone skimming this as a portfolio piece or before an interview, the ideas exercised here are: streaming API responses via generators/iterators, stateless-vs-stateful API design (why chat history has to be managed explicitly, and why a RAG call intentionally opts back out of it), the extract → chunk → embed → index → retrieve pipeline that underlies every production RAG system, cosine similarity and vector search implemented from first principles, prompt engineering to force grounding and a verbatim refusal message, deterministic citation attachment as a way to avoid trusting model self-reporting, the precision/recall gap that motivates hybrid search and reranking, and a layered architecture (client / service / model / retrieval / UI) that keeps each concern independently testable and replaceable.
