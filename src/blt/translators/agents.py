"""
Lyrics Translators
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from langchain_ollama import ChatOllama

from .models import LyricTranslation, MusicConstraints, SoramimiTranslation
from .analyzer import LyricsAnalyzer
from .validators import ConstraintValidator, SoramimiValidator
from .configs import LyricsTranslationAgentConfig, SoramimiTranslationAgentConfig
from .graphs import build_soramimi_mapping_graph

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

        # Configure Ollama model
        model = OpenAIChatModel(
            model_name=self.config.model,
            provider=OllamaProvider(base_url=self.config.ollama_base_url),
        )

        # Initialize agent (without system prompt yet)
        self.agent = Agent(
            model=model,
            output_type=LyricTranslation,
            system_prompt="",  # Will be set dynamically
            model_settings={"format": "json"},
        )

        # Register tools (this populates tool metadata in config)
        self.config.register_tools(self.agent, self.analyzer, self.validator)

        # Now set system prompt based on registered tools
        self.agent._system_prompt = self.config.get_system_prompt()

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

        # Reset stats
        self.config.reset_stats()

        # Extract constraints
        if constraints is None:
            constraints = self.analyzer.extract_constraints(source_lyrics, source_lang)

        # Build prompt (dynamically generated)
        user_prompt = self.config.get_user_prompt(
            source_lyrics,
            source_lang,
            target_lang,
            constraints.syllable_counts,
            constraints.rhyme_scheme or "",
            constraints.syllable_patterns,
        )

        # Translate
        logger.info("🚀 Starting translation...")
        result = self.agent.run_sync(user_prompt)
        translation = result.output
        logger.info("✅ Translation completed")

        # Calculate metrics
        translation.syllable_counts = [
            self.analyzer.count_syllables(line, target_lang)
            for line in translation.translated_lines
        ]
        translation.rhyme_endings = [
            self.analyzer.extract_rhyme_ending(line, target_lang)
            for line in translation.translated_lines
        ]
        translation.syllable_patterns = self.analyzer.get_syllable_patterns(
            translation.translated_lines, target_lang
        )
        translation.tool_call_stats = self.config.get_stats()

        # Validate
        validation = self.validator.validate(translation, constraints, target_lang)

        # Display
        elapsed = time.time() - start_time
        if validation.passed:
            print(f"✓ All constraints satisfied ({elapsed:.1f}s)")
        else:
            print(f"⚠ Score: {validation.score:.0%} ({elapsed:.1f}s)")

        if self.config.get_stats():
            print("\n📊 Tool Calls:")
            for name, count in sorted(self.config.get_stats().items()):
                print(f"   {name}: {count}")

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
        self.graph = build_soramimi_mapping_graph(self.analyzer, self.validator, self.llm)

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
                line.strip() for line in source_lyrics.strip().split("\n") if line.strip()
            ]
        else:
            source_lines = [line.strip() for line in source_lyrics if line.strip()]

        # Handle Chinese -> Pinyin
        chinese_lang_codes = ["cmn", "zh", "zh-cn", "zh-tw"]
        if source_lang.lower() in chinese_lang_codes:
            logger.info("   Converting Chinese to pinyin")
            source_lines = [self.analyzer._chinese_to_pinyin(line) for line in source_lines]
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
        initial_state = {
            "source_lines": source_lines,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_phonemes": [],
            "phoneme_mapping": {},
            "mapping_scores": {},
            "soramimi_lines": None,
            "source_ipa": None,
            "target_ipa": None,
            "similarity_scores": None,
            "overall_similarity": None,
            "best_mapping": None,
            "best_lines": None,
            "best_scores": None,
            "best_ipas": None,
            "attempt": 1,
            "max_attempts": self.config.max_retries,
            "threshold": self.config.similarity_threshold,
            "messages": [],
        }

        # Run graph
        print("   🗺️  Starting mapping-based soramimi creation...")
        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": 100,  # Allow up to 100 node executions
                "run_name": "SoramimiTranslation",  # LangSmith run name
            }
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
        print(f"\n   ✓ Completed in {elapsed:.1f}s - Similarity: {overall_similarity:.1%}")

        # Auto-save
        if self.config.auto_save:
            self._save(translation, source_lang, target_lang)

        return translation

    def _save(self, translation: SoramimiTranslation, src: str, tgt: str):
        """Save translation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"soramimi_mapping_{src}_to_{tgt}_{timestamp}.{self.config.save_format}"
        path = os.path.join(self.config.save_dir, filename)
        translation.save(path, format=self.config.save_format)
        print(f"   💾 Saved: {path}")
