import logging
import re
import time

from app.core import config
from app.llm.client import client

logger = logging.getLogger("docmind")

_RETRY_DELAY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*s")


def _is_rate_limit_error(error: Exception) -> bool:
    return getattr(error, "code", None) == 429


def _extract_retry_delay(error: Exception) -> float | None:
    """Pull the server-suggested retry delay (seconds) out of a 429
    RESOURCE_EXHAUSTED error's details, if present.

    google-genai's ClientError carries the parsed error JSON on
    `.details`, which for a quota error includes a RetryInfo block like
    {"@type": "...RetryInfo", "retryDelay": "55s"}. This is best-effort --
    the exact shape isn't a documented guarantee -- so returning None (and
    falling back to our own backoff) is the safe default, not an error.
    """
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None

    for item in details.get("error", {}).get("details", []):
        if str(item.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY_PATTERN.search(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))

    return None


def complete(prompt: str, max_retries: int = 5) -> str:
    """Stateless, single-shot completion -- no chat history involved.

    Used by RagService: each RAG call already carries its own full context
    (the retrieved chunks) baked into the prompt, so it shouldn't also pile
    into an ongoing multi-turn conversation the way ask()/stream() do.

    Retries on 429 (rate limit) errors, waiting for the server's suggested
    delay when it provides one. This matters in practice, not just in
    theory: the Gemini free tier caps requests at 5 per minute (as of
    writing), and the evaluation harness makes roughly two calls per
    question (answer + LLM-as-a-judge) across three retrieval modes --
    a burst well past that limit is the normal case for scripts/run_eval.py,
    not an edge case to shrug off.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.MODEL_NAME,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                delay = _extract_retry_delay(e) or (2**attempt) * 5
                logger.warning(
                    "Rate limited (attempt %d/%d) -- waiting %.0fs before retrying",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue

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
