MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048

# Ingestion / chunking (Phase 2)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embeddings (Phase 3)
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
TOP_K = 5

# Retrieval quality (Phase 4)
# A chunk must score at least this well (cosine similarity, 0-1) to be
# considered relevant at all. This is a starting value, not a calibrated
# one -- run scripts/test_search.py against your real documents and look at
# the printed scores for genuinely relevant vs. irrelevant queries before
# trusting this number. Too high and real answers get rejected as "not
# found"; too low and irrelevant chunks still reach the LLM.
SIMILARITY_THRESHOLD = 0.70

# Persistent vector storage (Phase 4, Step 2)
CHROMA_PERSIST_DIR = "data/chroma"
CHROMA_COLLECTION_NAME = "docmind"

# Cross-encoder reranking (Phase 4, Step 4/5)
# ms-marco-MiniLM-L-6-v2 is the standard small, fast cross-encoder trained
# specifically for passage reranking -- a reasonable default, not
# necessarily the most accurate option available.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# How many candidates to pull from the base retriever before reranking
# narrows them down to TOP_K. Wider than TOP_K on purpose: a chunk a
# cross-encoder would rate highly might only be ranked, say, #15 by
# dense/BM25/RRF -- it needs a chance to be *in* the candidate pool before
# reranking can promote it.
RERANK_CANDIDATE_POOL = 20
