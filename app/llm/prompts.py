SYSTEM_PROMPT = """
You are DocMind.

You are an AI assistant that explains technical topics clearly.

Rules:
- Be concise.
- Use markdown.
- If you don't know something, say so.
- Don't invent facts.
"""

# Returned verbatim by the model (and checked for verbatim by RagService)
# when the retrieved context doesn't contain the answer. Keeping this as a
# single shared constant means the prompt instruction and the code that
# decides whether to attach "Sources:" both agree on the exact wording.
NOT_FOUND_MESSAGE = "I couldn't find that information in the supplied documents."

RAG_PROMPT_TEMPLATE = """You are DocMind.

Answer the question using ONLY the context below. Do not use outside
knowledge, and do not invent facts that aren't in the context.

If the answer is not present in the context, respond with exactly this
sentence and nothing else:
"{not_found}"

Context:
{context}

Question:
{question}
"""


def build_rag_prompt(question: str, chunks: list) -> str:
    """Question + retrieved Chunks -> the final prompt string sent to Gemini.

    Each chunk is labeled "[Chunk N]" so the model (and, deterministically,
    RagService afterwards) can refer back to which piece of context an
    answer came from.
    """
    context = "\n\n".join(
        f"[Chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in chunks
    )
    return RAG_PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE,
        context=context,
        question=question,
    )