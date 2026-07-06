import fitz  # PyMuPDF


def load_pdf_text(path: str) -> str:
    """Extract raw text from a PDF file, one page at a time.

    PyMuPDF opens the file, and get_text() pulls the text layer off each
    page. This is plain extraction only -- no cleanup, no chunking. Real
    PDFs are messy (multi-column layouts, headers/footers, tables), so
    don't be surprised if the output isn't perfectly ordered prose.
    """
    doc = fitz.open(path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    return "\n".join(pages)
