"""Manual test for Phase 2: run the ingestion pipeline against a PDF and
print the same summary the roadmap expects (Loaded / Characters / Chunks).

Usage:
    python scripts/test_pipeline.py data/documents/resume.pdf
"""

import os
import sys

# Allow running as `python scripts/test_pipeline.py` from anywhere by putting
# the project root (one level up from scripts/) on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pipeline import process_pdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_pipeline.py <path-to-pdf>")
        sys.exit(1)

    path = sys.argv[1]
    document, chunks = process_pdf(path)

    print(f"Loaded {document.name}")
    print(f"Characters: {len(document.text)}")
    print(f"Chunks: {len(chunks)}")

    if chunks and chunks[0].embedding:
        dims = len(chunks[0].embedding)
        print(f"Embedding dimensions: {dims}")
        print(f"First 5 values of chunk 0: {chunks[0].embedding[:5]}")


if __name__ == "__main__":
    main()
