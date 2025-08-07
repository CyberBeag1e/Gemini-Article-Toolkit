from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
import logging, textwrap
from typing import List, Dict, Any, Optional
from .client import gemini
from .config import MODEL_SUMMARIZE, MODEL_QA, MAX_CHARS

def _clean(text: str) -> str:
    return textwrap.dedent(text).strip()

@dataclass
class Summary:
    headline:   str
    summary:    str
    topics:     List[str]
    keywords:   List[str]

@dataclass
class Sentences:
    sent_lst: List[str]

@dataclass
class Answer:
    answer: str
    evidence: List[str]

@dataclass
class NewsProcessor:
    article: str
    entity: Optional[str] = None

    def summarize(self, max_length: int = 100) -> Dict[str, Any]:
        """
        Summarize the article into
        - A headline: within 10 words
        - A short summary: several concise sentences within a certain length. The total length is controlled by the parameter `max_length`.
        - A list of topics: 2-5 topics
        - A list of keywords: 3-6 keywords that appears in the article

        Args:
            `max_length` (`int`): maximum length of the summary.

        Returns:
            `summary` (`dict`): a dictionary containing "headline", "summary", "topics", "keywords" as keys.
        """

        prompt = _clean(f"""
        Summarize the following article into the following components:
        "headline": 10 words max;
        "summary": several concise sentences within {max_length} words in total;
        "topics": list of 2-5 one-/two-word topics;
        "keywords": list of 3-6 keywords that appear in the article.

        Use the article below as the sole source of truth:

        ARTICLE:
        ```txt
        {self.article[:MAX_CHARS]}
        ```
        """)

        raw = gemini.safe_generate(MODEL_SUMMARIZE, 
                                   prompt, 
                                   config = {
                                       "response_mime_type": "application/json",
                                       "response_schema": Summary,
                                   }, 
                                   validate = lambda d: isinstance(d, Summary))

        return self._transform(raw)
    
    def highlight_entity(self) -> List[str]:
        """
        Find facts about the specified `entity` (a person, a company, or a character, etc.) in the article. The facts are the sentences that directly/indirectly refer to the entity

        Returns:
            `sentences` (`dict`): a dictionary containing "sent_lst" as the key, where `sent_lst` will be a list of string.
        """

        if not self.entity:
            return []
        
        prompt = _clean(f"""
        Given the article, and an entity name {self.entity}.
        Return a list of sentences that refer to the entity in the article, both direct and indirect matches (e.g. pronouns, nicknames, abbreviations, subsidiaries, etc.).

        ARTICLE:
        ```txt
        {self.article[:MAX_CHARS]}
        ```
        """)

        results = gemini.safe_generate(MODEL_QA, 
                                       prompt, 
                                       config = {
                                            "response_mime_type": "application/json",
                                            "response_schema": Sentences,
                                       }, 
                                       validate = lambda d: isinstance(d, Sentences))

        return self._transform(results)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the article. 

        Args:
            `question` (`str`): a question.

        Returns:
            `answer` (`dict`): a dictionary with "answer" and "evidence" as keys, where `evidence` will be a list of keywords that justify the answer in the article.
        """

        if not question:
            return {}
        
        question = question if question.endswith("?") else question + "?"
        prompt = _clean(f"""
        Read the article, and answer the user's question.
        Return a JSON dict with keys:
        "answer": <short answer>,
        "evidence": [list of KEYWORDS IN the article that justify the answer]
                        
        ARTICLE:
        ```txt
        {self.article[:MAX_CHARS]}
        ```

        QUESTION:
        {question}
        """)

        raw = gemini.safe_generate(MODEL_QA, 
                                   prompt, 
                                   config = {
                                        "response_mime_type": "application/json",
                                        "response_schema": Answer,
                                   },  
                                   validate = lambda d: isinstance(d, Answer))
        
        return self._transform(raw)
    
    def _transform(self, output: Any) -> Dict[str, Any]:
        """
        Transform the `dataclass` into a `dict`.

        Args:
            `output` (`DataClassInstance`): a class instance wrapped by `dataclass`
        
        Returns:
            `transformed_output` (`dict`): a dictionary with attribute names as keys.
        """
        if is_dataclass(output):
            return asdict(output)
        else:
            logging.warning(f"Object {output} is not a dataclass instance.")
            return output