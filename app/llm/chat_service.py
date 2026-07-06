import logging

from app.core import config
from app.llm.client import client

logger = logging.getLogger("docmind")


def complete(prompt: str) -> str:
    """Stateless, single-shot completion -- no chat history involved.

    Used by RagService: each RAG call already carries its own full context
    (the retrieved chunks) baked into the prompt, so it shouldn't also pile
    into an ongoing multi-turn conversation the way ask()/stream() do.
    """
    try:
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        logger.error("API error: %s", e)
        raise


class ChatService:
    def __init__(self):
        # client.chats.create() gives us a session object that remembers
        # every user/model turn internally (a `history` list under the
        # hood). Each call to send_message() appends to that history and
        # sends the *whole* conversation so far, not just the latest line.
        self.chat = client.chats.create(
            model=config.MODEL_NAME,
            config={
                "temperature": config.TEMPERATURE,
                "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            },
        )

    def ask(self, message: str) -> str:
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            logger.error("API error: %s", e)
            raise

    def stream(self, message: str):
        try:
            response = self.chat.send_message_stream(message)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("API error: %s", e)
            raise

    def history(self):
        """Expose the underlying turn history, mostly useful for debugging."""
        return self.chat.get_history()
