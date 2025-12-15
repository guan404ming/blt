"""
Lyrics Translators
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from .models import LyricTranslation, MusicConstraints, SoramimiTranslation
from .analyzer import LyricsAnalyzer
from .validators import ConstraintValidator, SoramimiValidator
from .configs import LyricsTranslationAgentConfig, SoramimiTranslationAgentConfig
from .graphs import (
    build_lyrics_translation_graph,
    build_soramimi_mapping_graph,
    create_lyrics_translation_initial_state,
    create_soramimi_mapping_initial_state,
)

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LyricsTranslationAgent:
    """Lyrics translator with unified configuration"""

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

        # Core components
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
        Translate lyrics

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
            self._save(translation, source_lang, target_lang)

        return translation

    def _save(self, translation: LyricTranslation, src: str, tgt: str):
        """Save translation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"translation_{src}_to_{tgt}_{timestamp}.{self.config.save_format}"
        path = os.path.join(self.config.save_dir, filename)
        translation.save(path, format=self.config.save_format)
        print(f"\n✓ Saved: {path}")


# ==================== SORAMIMI TRANSLATOR ====================


class SoramimiTranslationAgent:
    """
    Mapping-based Soramimi translator

    Workflow:
    1. Extract unique phonemes from source
    2. Build phoneme -> target character mapping
    3. Refine mapping iteratively
    4. Apply mapping to generate soramimi
    """

    def __init__(
        self,
        config: Optional[SoramimiTranslationAgentConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        self.config = config or SoramimiTranslationAgentConfig()
        self.analyzer = analyzer or LyricsAnalyzer()
        self.validator = SoramimiValidator(
            self.analyzer,
            self.config.similarity_threshold,
        )

        # Configure LangSmith
        if self.config.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project
            if not os.getenv("LANGCHAIN_API_KEY"):
                logger.warning(
                    "LangSmith tracing enabled but LANGCHAIN_API_KEY not set."
                )

        # Create LLM
        self.llm = ChatOllama(
            model=self.config.model,
            base_url=self.config.ollama_base_url.replace("/v1", ""),
            format="json",
            temperature=0.7,
        )

        # Build graph
        self.graph = build_soramimi_mapping_graph(
            self.analyzer, self.validator, self.llm
        )

    def translate(
        self,
        source_lyrics: str | list[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> SoramimiTranslation:
        """
        Create soramimi translation using mapping approach

        Args:
            source_lyrics: Source lyrics
            source_lang: Source language
            target_lang: Target language

        Returns:
            SoramimiTranslation
        """
        start_time = time.time()

        source_lang = source_lang or self.config.default_source_lang
        target_lang = target_lang or self.config.default_target_lang

        # Parse lines
        if isinstance(source_lyrics, str):
            source_lines = [
                line.strip()
                for line in source_lyrics.strip().split("\n")
                if line.strip()
            ]
        else:
            source_lines = [line.strip() for line in source_lyrics if line.strip()]

        # Handle Chinese -> Pinyin
        chinese_lang_codes = ["cmn", "zh", "zh-cn", "zh-tw"]
        if source_lang.lower() in chinese_lang_codes:
            logger.info("   Converting Chinese to pinyin")
            source_lines = [
                self.analyzer._chinese_to_pinyin(line) for line in source_lines
            ]
            source_lang = "en-us"

        # Early return if same language
        if source_lang == target_lang:
            return SoramimiTranslation(
                soramimi_lines=source_lines,
                source_ipa=[],
                target_ipa=[],
                similarity_scores=[1.0] * len(source_lines),
                overall_similarity=1.0,
                reasoning="Same language",
            )

        # Initialize state
        initial_state = create_soramimi_mapping_initial_state(
            source_lines,
            source_lang,
            target_lang,
            self.config.max_retries,
            self.config.similarity_threshold,
        )

        # Run graph
        print("   🗺️  Starting mapping-based soramimi creation...")
        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": 100,  # Allow up to 100 node executions
                "run_name": "SoramimiTranslation",  # LangSmith run name
            },
        )

        # Build result
        best_lines = final_state["best_lines"]
        best_scores = final_state["best_scores"]
        best_ipas = final_state["best_ipas"]

        if not best_lines or any(line is None for line in best_lines):
            raise RuntimeError("Failed to generate soramimi translation")

        overall_similarity = sum(best_scores) / len(best_scores) if best_scores else 0

        translation = SoramimiTranslation(
            soramimi_lines=list(best_lines),
            source_ipa=[ipa[0] for ipa in best_ipas],
            target_ipa=[ipa[1] for ipa in best_ipas],
            similarity_scores=list(best_scores),
            overall_similarity=overall_similarity,
            reasoning=f"Mapping-based: {len(final_state['phoneme_mapping'])} phonemes mapped over {final_state['attempt']} attempts",
        )

        # Display
        elapsed = time.time() - start_time
        print(
            f"\n   ✓ Completed in {elapsed:.1f}s - Similarity: {overall_similarity:.1%}"
        )

        # Auto-save
        if self.config.auto_save:
            self._save(translation, source_lang, target_lang)

        return translation

    def _save(self, translation: SoramimiTranslation, src: str, tgt: str):
        """Save translation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"soramimi_mapping_{src}_to_{tgt}_{timestamp}.{self.config.save_format}"
        )
        path = os.path.join(self.config.save_dir, filename)
        translation.save(path, format=self.config.save_format)
        print(f"   💾 Saved: {path}")
