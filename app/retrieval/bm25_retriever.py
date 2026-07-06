import re

from rank_bm25 import BM25Okapi

from app.core import config
from app.models.chunk import Chunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, then extract runs of alphanumeric characters as tokens.

    BM25 matches tokens *exactly*, so punctuation attached to a word is not
    a cosmetic detail -- splitting on whitespace alone turns "PyMuPDF,"
    (comma attached, as it appears mid-sentence in real text) into the
    token "pymupdf," which will never match a query's "pymupdf" token.
    Regex-extracting word characters strips that punctuation instead of
    silently breaking every term that happens to be followed by a comma or
    period, which is most of them.

    Still intentionally simple otherwise (no stemming/stopword removal) --
    that's a real limitation, not this one.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Question -> relevant Chunks, via keyword (term-frequency) matching
    instead of semantic similarity.

    Complements Retriever (dense/embedding search): embeddings represent
    overall meaning, so a query for an exact string like "PyMuPDF" might
    not score highly by cosine similarity if the surrounding context
    doesn't emphasize it -- but BM25 finds it immediately, because it's
    just counting term overlap weighted by how rare/informative each term
    is across the corpus.

    Unlike Retriever, this needs the *entire* corpus up front (to compute
    term frequencies across all chunks), rather than searching a vector
    store per query -- hence the constructor takes chunks directly instead
    of a store.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._index = BM25Okapi([_tokenize(chunk.text) for chunk in chunks]) if chunks else None

    def retrieve(self, question: str, top_k: int = None) -> list[tuple[float, Chunk]]:
        top_k = top_k or config.TOP_K

        if self._index is None:
            return []

        scores = self._index.get_scores(_tokenize(question))
        ranked = sorted(zip(scores, self.chunks), key=lambda pair: pair[0], reverse=True)

        # A score of exactly 0 means no term overlap at all -- BM25's own
        # signal that a chunk isn't a match, not just a weak one. Dropping
        # these (rather than returning the "best" zero-score chunks) is
        # what lets BM25 contribute nothing to a query it has no lexical
        # basis for, the keyword-search equivalent of Retriever's
        # similarity threshold.
        return [(score, chunk) for score, chunk in ranked[:top_k] if score > 0]
