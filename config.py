import os
from typing import Final

"""
Set the GEMINI_API_KEY as your environment variable first.
Change the model to "gemini-2.5-pro" for advanced features.
"""

GEMINI_API_KEY: Final[str] = os.environ["GEMINI_API_KEY"]

MODEL_SUMMARIZE: Final[str] = "gemini-2.5-flash"
MODEL_QA       : Final[str] = "gemini-2.5-flash"
MAX_CHARS      : Final[str] = 24_000