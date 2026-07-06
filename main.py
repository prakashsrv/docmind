import logging
import time

from app.llm.chat_service import ChatService
from app.ui.console import (
    ask_user,
    print_error,
    print_goodbye,
    print_welcome,
    stream_response,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
logger = logging.getLogger("docmind")

service = ChatService()

print_welcome()

while True:
    question = ask_user()

    if question.lower() == "exit":
        print_goodbye()
        break

    try:
        logger.info("Sending request")
        start = time.perf_counter()
        stream_response(service.stream(question))
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Response received in %.0f ms", elapsed_ms)
    except Exception as e:
        print_error(str(e))
