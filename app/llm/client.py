import logging
import os

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger("docmind")

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

logger.info("Loading Gemini client")
client = genai.Client(api_key=api_key)