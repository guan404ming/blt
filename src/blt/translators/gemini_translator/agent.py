"""Gemini Lyrics Translation Agent - Simple pattern with validate-retry flow

Uses LangChain's ChatGoogleGenerativeAI with 3 constraint validation tools.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import GeminiTranslatorConfig
from .models import GeminiTranslation
from .validator import Validator
from .graph import build_graph, create_initial_state

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiTranslationAgent:
    """Simple Gemini-based lyrics translator with validate-retry flow

    Uses LangChain's ChatGoogleGenerativeAI with shared constraint tools:
    - count_syllables (from shared)
    - extract_rhyme_ending (from shared)
    - detect_rhyme_scheme (from shared)
    - get_syllable_patterns (from shared)

    Tools are bound to LLM in graph.py during initialization.
    """

    def __init__(self, config: Optional[GeminiTranslatorConfig] = None):
        """Initialize Gemini translation agent

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or GeminiTranslatorConfig()

        # Get API key from config or environment
        # Check in order: config, GOOGLE_API_KEY, GEMINI_API_KEY
        api_key = (
            self.config.api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Google API key not found. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key in GeminiTranslatorConfig"
            )

        # Create LangChain Gemini LLM (tools bound in graph.py)
        llm = ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=api_key,
            temperature=self.config.temperature,
        )

        # Create validator for constraint verification
        self.validator = Validator()

        # Build graph (binds constraint tools to LLM)
        self.graph = build_graph(llm, self.validator)

    def translate(
        self,
        source_lyrics: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        syllable_counts: Optional[list[int]] = None,
        rhyme_scheme: Optional[str] = None,
        syllable_patterns: Optional[list[list[int]]] = None,
    ) -> GeminiTranslation:
        """Translate lyrics with constraint validation and retry

        Graph flow:
        1. Generate translation (with constraint tools available)
        2. Validate against constraints (syllables, rhyme, patterns)
        3. If not valid and attempts < 3:
           - Wait 3 seconds
           - Retry generation

        Args:
            source_lyrics: Source lyrics
            source_lang: Source language (uses config default if None)
            target_lang: Target language (uses config default if None)
            syllable_counts: Required syllable counts per line
            rhyme_scheme: Required rhyme scheme (e.g., "ABCDAECDD")
            syllable_patterns: Required syllable patterns per line

        Returns:
            GeminiTranslation with results
        """
        start_time = time.time()

        # Use defaults
        source_lang = source_lang or self.config.default_source_lang
        target_lang = target_lang or self.config.default_target_lang

        print("   🚀 Starting Gemini translation with validate-retry flow...")

        # Create initial state
        initial_state = create_initial_state(
            source_lyrics=source_lyrics,
            source_lang=source_lang,
            target_lang=target_lang,
            syllable_counts=syllable_counts or [],
            rhyme_scheme=rhyme_scheme or "",
            syllable_patterns=syllable_patterns or [],
        )

        # Run graph
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 10, "run_name": "GeminiTranslation"},
        )

        # Build result
        translation = GeminiTranslation(
            translated_lines=final_state.get("translated_lines") or [],
            syllable_counts=final_state.get("translation_syllable_counts") or [],
            rhyme_scheme=final_state.get("translation_rhyme_scheme") or "",
            validation=final_state.get("validation_details") or {},
            reasoning=final_state.get("reasoning") or "",
        )

        # Display results
        elapsed = time.time() - start_time
        if final_state.get("validation_passed"):
            print(
                f"\n   ✓ Translation validated ({elapsed:.1f}s, {final_state['attempt']} attempt(s))"
            )
        else:
            print(
                f"\n   ⚠ Translation incomplete after {final_state['attempt']} attempt(s) "
                f"({elapsed:.1f}s)"
            )

        # Auto-save
        if self.config.auto_save:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"gemini_translation_{source_lang}_{target_lang}_{timestamp}."
                f"{self.config.save_format}"
            )
            file_path = save_dir / filename
            translation.save(str(file_path), format=self.config.save_format)
            logger.info(f"Translation saved to {file_path}")

        return translation
