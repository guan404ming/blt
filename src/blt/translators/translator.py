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

from .models import LyricTranslation, MusicConstraints, SoramimiTranslation
from .analyzer import LyricsAnalyzer
from .validator import ConstraintValidator, SoramimiValidator
from .config import TranslatorConfig, SoramimiConfig

logger = logging.getLogger(__name__)


class LyricsTranslator:
    """Lyrics translator with unified configuration"""

    def __init__(
        self,
        config: Optional[TranslatorConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        """
        Initialize translator

        Args:
            config: Configuration (uses defaults if None)
            analyzer: Lyrics analyzer (creates new if None)
        """
        self.config = config or TranslatorConfig()

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


class SoramimiTranslator:
    """
    Soramimi (phonetic) translator that creates sound-alike translations

    The translator:
    1. Takes source lyrics and target language
    2. Uses IPA analysis to compare phonetic similarity
    3. Iteratively improves translations to maximize sound similarity
    """

    def __init__(
        self,
        config: Optional[SoramimiConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        """
        Initialize soramimi translator

        Args:
            config: Configuration (uses defaults if None)
            analyzer: Lyrics analyzer (creates new if None)
        """
        self.config = config or SoramimiConfig()

        # Core components
        self.analyzer = analyzer or LyricsAnalyzer()
        self.validator = SoramimiValidator(
            self.analyzer,
            self.config.similarity_threshold,
        )

        # Configure Ollama model
        model = OpenAIChatModel(
            model_name=self.config.model,
            provider=OllamaProvider(base_url=self.config.ollama_base_url),
        )

        # Initialize agent
        self.agent = Agent(
            model=model,
            output_type=SoramimiTranslation,
            system_prompt="",
            model_settings={"format": "json"},
        )

        # Register tools
        self.config.register_tools(self.agent, self.analyzer, self.validator)

        # Set system prompt
        self.agent._system_prompt = self.config.get_system_prompt()

    def translate(
        self,
        source_lyrics: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> SoramimiTranslation:
        """
        Create soramimi translation

        Args:
            source_lyrics: Source lyrics
            source_lang: Source language (uses config default if None)
            target_lang: Target language (uses config default if None)

        Returns:
            SoramimiTranslation with results
        """
        start_time = time.time()

        # Defaults
        source_lang = source_lang or self.config.default_source_lang
        target_lang = target_lang or self.config.default_target_lang

        # Reset stats
        self.config.reset_stats()

        # Parse source lines and store in config for tools
        source_lines = [
            line.strip() for line in source_lyrics.strip().split("\n") if line.strip()
        ]
        self.config._source_lines = source_lines
        self.config._source_lang = source_lang
        self.config._target_lang = target_lang

        # Update system prompt with correct languages
        self.agent._system_prompt = self.config.get_system_prompt()

        # Build initial prompt
        user_prompt = self.config.get_user_prompt(
            source_lyrics, source_lang, target_lang
        )

        # Retry loop - keep track of best result
        logger.info("   Starting soramimi creation...")
        best_translation = None
        best_validation = None
        best_similarity = 0.0

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Generate soramimi
                result = self.agent.run_sync(user_prompt)
                translation = result.output

                # Calculate metrics
                validation = self.validator.compare_ipa(
                    source_lines,
                    translation.soramimi_lines,
                    source_lang,
                    target_lang,
                )

                # Update translation with calculated values
                translation.source_ipa = validation["source_ipas"]
                translation.target_ipa = validation["target_ipas"]
                translation.similarity_scores = validation["similarities"]
                translation.overall_similarity = validation["overall_similarity"]

                # Keep best result
                if validation["overall_similarity"] > best_similarity:
                    best_similarity = validation["overall_similarity"]
                    best_translation = translation
                    best_validation = validation
                    logger.info(f"   ✓ New best: {best_similarity:.1%} (attempt {attempt})")

                # Check if all lines pass
                if validation["passed"]:
                    logger.info(f"   ✓ All lines passed on attempt {attempt}")
                    break

            except Exception as e:
                logger.warning(f"   Attempt {attempt} failed with error: {e}")
                if attempt == self.config.max_retries:
                    # On last attempt, if we have a best result, use it; otherwise raise
                    if best_translation is not None:
                        logger.warning("   Using best result from previous attempts")
                        break
                    raise
                # Otherwise continue to next retry
                user_prompt = f"""Previous attempt had errors. Please try again.

Original {source_lang} lyrics:
{source_lyrics}

Remember: Return ONLY valid JSON with "soramimi_lines" and "reasoning" fields.
Create {target_lang} soramimi that SOUNDS like the {source_lang}."""
                continue

            # If not last attempt and not passed, prepare feedback for retry
            if attempt < self.config.max_retries and not validation["passed"]:
                passed_lines = []
                failed_lines = []

                for i, (score, line, src) in enumerate(
                    zip(validation["similarities"], translation.soramimi_lines, source_lines), 1
                ):
                    if score >= self.config.similarity_threshold:
                        passed_lines.append(f"{i}. {line} ({score:.1%} PASS)")
                    else:
                        failed_lines.append(f"{i}. {line} ({score:.1%} FAIL)")

                feedback_parts = []
                if passed_lines:
                    feedback_parts.append("KEEP UNCHANGED:\n" + "\n".join(passed_lines))
                if failed_lines:
                    feedback_parts.append("IMPROVE THESE:\n" + "\n".join(failed_lines))

                feedback = "\n\n".join(feedback_parts)
                logger.info(f"   Attempt {attempt}: {validation['overall_similarity']:.1%} overall")

                # Build simple retry prompt
                user_prompt = f"""Attempt {attempt}: {validation['overall_similarity']:.1%} overall. Target: {self.config.similarity_threshold:.0%}+

{feedback}

Keep passing lines UNCHANGED. Only revise failing lines to match pronunciation better."""
            else:
                if attempt == self.config.max_retries:
                    logger.info(f"   Max retries ({self.config.max_retries}) reached")

        # Use best result
        translation = best_translation
        validation = best_validation

        logger.info(f"   Soramimi creation completed - using best result: {best_similarity:.1%}")

        # Update final stats
        translation.tool_call_stats = self.config.get_stats()

        # Display results
        elapsed = time.time() - start_time
        if validation["passed"]:
            print(
                f"   All lines meet similarity threshold "
                f"({validation['overall_similarity']:.1%} overall, {elapsed:.1f}s)"
            )
        else:
            print(
                f"   Overall similarity: {validation['overall_similarity']:.1%} "
                f"({elapsed:.1f}s)"
            )

        if self.config.get_stats():
            print("\n   Tool Calls:")
            for name, count in sorted(self.config.get_stats().items()):
                print(f"      {name}: {count}")

        # Auto-save
        if self.config.auto_save:
            self._save(translation, source_lang, target_lang)

        return translation

    def _save(self, translation: SoramimiTranslation, src: str, tgt: str):
        """Save translation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"soramimi_{src}_to_{tgt}_{timestamp}.{self.config.save_format}"
        path = os.path.join(self.config.save_dir, filename)
        translation.save(path, format=self.config.save_format)
        print(f"\n   Saved: {path}")
