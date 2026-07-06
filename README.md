# DocMind

DocMind is a terminal-based AI assistant built from scratch, in phases, as a way to learn how a real Retrieval-Augmented Generation (RAG) system is put together — not by installing a framework, but by building each piece: an LLM client, a chat service, a PDF ingestion pipeline, and a semantic search engine.

This document explains both **what was built** and **why it works the way it does**, so it can be read as a project write-up or as a study guide for the concepts involved.

---

## What DocMind does today

Right now, DocMind is two things living in the same codebase:

1. A streaming, multi-turn terminal chatbot backed by Google's Gemini API.
2. A standalone document-ingestion and semantic-search pipeline: PDF → text → chunks → embeddings → an in-memory vector store you can query by meaning, not keywords.

These two halves aren't wired together yet — that's the next phase (feeding search results into the chatbot as context, which is what turns a chatbot into "RAG"). Everything below reflects what's actually implemented.

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
│   └── ui/
│       └── console.py            # Rich-based terminal rendering
├── scripts/
│   ├── test_pipeline.py          # Manual test: ingest + embed a PDF
│   └── test_search.py            # Manual test: ingest, then semantic search
├── data/
│   └── documents/                # Source PDFs (gitignored, except .gitkeep)
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

## What's next

The two halves described above — the chatbot and the search pipeline — aren't connected yet. The next phase is exactly that connection, which is what actually makes this a RAG system rather than "a chatbot" and "a search demo" sitting side by side:

```
User question
      │
      ▼
EmbeddingService.embed(question)
      │
      ▼
VectorStore.search(query_vector)
      │
      ▼
Top-k relevant chunks
      │
      ▼
Inject chunks into the prompt sent to ChatService
      │
      ▼
Gemini answers using retrieved context, not just training data
```

Beyond that, planned directions include replacing `InMemoryVectorStore` with a persistent vector database (Chroma), applying the `SYSTEM_PROMPT` already defined in `prompts.py` to the chat session (currently written but not yet wired in), and evaluating chunk size/overlap empirically instead of by default values.

---

## Concepts this project demonstrates

For anyone skimming this as a portfolio piece or before an interview, the ideas exercised here are: streaming API responses via generators/iterators, stateless-vs-stateful API design (why chat history has to be managed explicitly), the extract → chunk → embed → index pipeline that underlies every production RAG system, cosine similarity and vector search implemented from first principles, and a layered architecture (client / service / model / UI) that keeps each concern independently testable and replaceable.
