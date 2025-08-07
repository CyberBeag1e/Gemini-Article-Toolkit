from __future__ import annotations
from google import genai
import logging, time
from typing import Any, Dict, List, Callable
from .config import GEMINI_API_KEY

class GeminiClient:
    """
    A tiny Gemini client for text generation.
    """
    def __init__(self) -> None:
        self._client = genai.Client(api_key = GEMINI_API_KEY)
    
    def generate(self, model: str, prompt: str | List[Dict[str, Any]], **kwargs) -> str:
        res = self._client.models.generate_content(
            model = model, 
            contents = prompt,
            **kwargs
        )

        return res
    
    def safe_generate(self, model: str, 
                            prompt: str | List[Dict[str, Any]],
                            *,
                            max_tries: int = 3,
                            backoff: float = 1.5,
                            validate: Callable[[Any], bool] | None = None,
                            **kwargs):
        
        """
        A modified generate method that support output validation, in case that the Gemini model returns invalid output. 
        In the application, all the generation tasks return JSON dict.
        """
        validate = validate or (lambda x: bool(x))
        for attempt in range(max_tries):
            try:
                raw = self.generate(model, prompt, **kwargs)
                
                # The output is set to be some pre-defined dataclass. 
                # raw.parsed gets the instantiated objects.
                if validate(raw.parsed):
                    return raw.parsed
                else:
                    raise ValueError("Validation failed.")
            
            except Exception as exc:
                logging.warning(f"Gemini attempt {attempt + 1:d}/{max_tries:d} failed: {exc}")

                if attempt + 1 == max_tries:
                    raise

                time.sleep(backoff ** attempt)

gemini = GeminiClient()