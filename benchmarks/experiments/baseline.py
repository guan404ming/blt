"""
Baseline Translator (No Agent)

Simple translation using LLM without constraint-aware refinement.
This serves as a baseline for comparison with the agent-based approach.
"""

from __future__ import annotations
from typing import TypedDict
from langchain_ollama import ChatOllama


class BaselineTranslation(TypedDict):
    """Result from baseline translation"""

    translated_lines: list[str]
    source_lines: list[str]
    source_lang: str
    target_lang: str
    model: str


class BaselineTranslator:
    """
    Simple translator without constraint awareness

    This translator uses a language model to translate lyrics
    without any constraint verification or refinement loop.
    It represents standard neural machine translation.
    """

    def __init__(
        self,
        model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        """
        Initialize baseline translator

        Args:
            model: Ollama model name
            base_url: Ollama API base URL
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model = model
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    def translate(
        self,
        source_lines: list[str],
        source_lang: str,
        target_lang: str,
    ) -> BaselineTranslation:
        """
        Translate lyrics without constraint awareness

        Args:
            source_lines: Original lyrics lines
            source_lang: Source language code (e.g., 'cmn', 'en-us', 'ja')
            target_lang: Target language code

        Returns:
            BaselineTranslation with translated lines
        """
        # Format source text
        source_text = "\n".join(source_lines)

        # Build prompt
        prompt = self._build_prompt(source_text, source_lang, target_lang)

        # Get translation
        response = self.llm.invoke(prompt)
        translated_text = response.content.strip()

        # Parse lines (match source line count)
        translated_lines = translated_text.split("\n")

        # Ensure same number of lines as source
        while len(translated_lines) < len(source_lines):
            translated_lines.append("")
        if len(translated_lines) > len(source_lines):
            translated_lines = translated_lines[: len(source_lines)]

        return BaselineTranslation(
            translated_lines=translated_lines,
            source_lines=source_lines,
            source_lang=source_lang,
            target_lang=target_lang,
            model=self.model,
        )

    def _build_prompt(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Build translation prompt"""

        lang_names = {
            "cmn": "Chinese",
            "en-us": "English",
            "ja": "Japanese",
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        return f"""Translate song lyrics from {source_name} to {target_name}.

Source lyrics:
{source_text}

Translated lyrics:"""

    def translate_batch(
        self,
        source_lyrics_list: list[list[str]],
        source_lang: str,
        target_lang: str,
    ) -> list[BaselineTranslation]:
        """
        Translate multiple lyrics in batch

        Args:
            source_lyrics_list: List of lyrics (each is a list of lines)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of BaselineTranslation results
        """
        return [
            self.translate(source_lines, source_lang, target_lang)
            for source_lines in source_lyrics_list
        ]
