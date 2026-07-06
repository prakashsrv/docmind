from app.llm.chat_service import complete
from app.llm.prompts import NOT_FOUND_MESSAGE, build_rag_prompt
from app.retrieval.retriever import Retriever


class RagService:
    """Question -> grounded answer with sources.

    Orchestrates the pieces without doing their work itself:
    Retriever finds context, prompts.build_rag_prompt() shapes it into a
    prompt, chat_service.complete() calls Gemini. This class's own job is
    just wiring those together and attaching citations.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = None) -> str:
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return NOT_FOUND_MESSAGE

        prompt = build_rag_prompt(question, chunks)
        answer = complete(prompt)

        # If the model followed instructions and admitted it doesn't know,
        # don't attach sources -- there's nothing to cite.
        if answer.strip().startswith(NOT_FOUND_MESSAGE):
            return answer

        # Built deterministically from the chunks we actually retrieved,
        # rather than trusting the model to self-report which chunks it
        # used -- the model can hallucinate a citation, but it can't change
        # which chunks we know we sent it.
        sources = ", ".join(f"Chunk {chunk.chunk_index}" for chunk in chunks)
        return f"{answer}\n\nSources: {sources}"
