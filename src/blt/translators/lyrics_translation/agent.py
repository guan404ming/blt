"""
Lyrics Translation Agent

Translates lyrics while preserving constraints (syllable count, rhyme scheme, etc).
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_ollama import ChatOllama

from ..shared import LyricsAnalyzer
from .config import LyricsTranslationAgentConfig
from .models import LyricTranslation, MusicConstraints
from .graph import (
    build_lyrics_translation_graph,
    create_lyrics_translation_initial_state,
)
from .validator import ConstraintValidator

load_dotenv()
logger = logging.getLogger(__name__)


class LyricsTranslationAgent:
    """Lyrics translator with constraint-based quality control"""

    def __init__(
        self,
        config: Optional[LyricsTranslationAgentConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        """
        Initialize translator

        Args:
            config: Configuration (uses defaults if None)
            analyzer: Lyrics analyzer (creates new if None)
        """
        self.config = config or LyricsTranslationAgentConfig()
        self.analyzer = analyzer or LyricsAnalyzer()
        self.validator = ConstraintValidator(self.analyzer)

        # Configure LangSmith
        if hasattr(self.config, "langsmith_tracing") and self.config.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = getattr(
                self.config, "langsmith_project", "blt"
            )
            if not os.getenv("LANGCHAIN_API_KEY"):
                logger.warning(
                    "LangSmith tracing enabled but LANGCHAIN_API_KEY not set."
                )

        # Create LLM
        self.llm = ChatOllama(
            model=self.config.model,
            base_url=self.config.ollama_base_url.replace("/v1", ""),
            temperature=0.7,
        )

        # Build graph
        self.graph = build_lyrics_translation_graph(
            self.analyzer, self.llm, self.config
        )

    def translate(
        self,
        source_lyrics: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        constraints: Optional[MusicConstraints] = None,
    ) -> LyricTranslation:
        """
        Translate lyrics with constraint verification

        Args:
            source_lyrics: Source lyrics
            source_lang: Source language (uses config default if None)
            target_lang: Target language (uses config default if None)
            constraints: Music constraints (auto-extracted if None)

        Returns:
            LyricTranslation with results
        """
        start_time = time.time()

        # Defaults
        source_lang = source_lang or self.config.default_source_lang
        target_lang = target_lang or self.config.default_target_lang

        # Extract constraints
        if constraints is None:
            constraints = self.analyzer.extract_constraints(source_lyrics, source_lang)

        # Initialize state
        initial_state = create_lyrics_translation_initial_state(
            source_lyrics, source_lang, target_lang, constraints
        )

        # Run graph
        print("   🚀 Starting lyrics translation...")
        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": 50,
                "run_name": "LyricsTranslation",
            },
        )

        # Build result
        translation = LyricTranslation(
            translated_lines=final_state.get("translated_lines") or [],
            reasoning=final_state.get("reasoning") or "",
            syllable_counts=final_state.get("translation_syllable_counts") or [],
            rhyme_endings=final_state.get("translation_rhyme_endings") or [],
            syllable_patterns=final_state.get("translation_syllable_patterns") or [],
        )

        # Validate using the validator
        validation = self.validator.validate(translation, constraints, target_lang)

        # Display
        elapsed = time.time() - start_time
        if validation.passed:
            print(f"\n   ✓ All constraints satisfied ({elapsed:.1f}s)")
        else:
            print(f"\n   ⚠ Score: {validation.score:.0%} ({elapsed:.1f}s)")

        # Auto-save
        if self.config.auto_save:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_{source_lang}_{target_lang}_{timestamp}.{self.config.save_format}"
            file_path = save_dir / filename
            translation.save(str(file_path), format=self.config.save_format)
            logger.info(f"Translation saved to {file_path}")

        return translation
