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
from .validators import ConstraintValidator, SoramimiValidator
from .configs import LyricsTranslationAgentConfig, SoramimiTranslationAgentConfig

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
    Soramimi (phonetic) translator that creates sound-alike translations

    The translator:
    1. Takes source lyrics and target language
    2. Uses IPA analysis to compare phonetic similarity
    3. Iteratively improves translations to maximize sound similarity
    """

    def __init__(
        self,
        config: Optional[SoramimiTranslationAgentConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        """
        Initialize soramimi translator

        Args:
            config: Configuration (uses defaults if None)
            analyzer: Lyrics analyzer (creates new if None)
        """
        self.config = config or SoramimiTranslationAgentConfig()

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

        # Retry loop - keep track of best result PER LINE
        logger.info("   Starting soramimi creation...")
        num_lines = len(source_lines)
        best_lines = [None] * num_lines
        best_line_scores = [0.0] * num_lines
        best_line_ipas = [("", "")] * num_lines  # (source_ipa, target_ipa) tuples

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

                # Keep best result PER LINE (not overall average)
                improved_lines = []
                for i, (score, line) in enumerate(
                    zip(validation["similarities"], translation.soramimi_lines)
                ):
                    if score > best_line_scores[i]:
                        old_score = best_line_scores[i]
                        best_line_scores[i] = score
                        best_lines[i] = line
                        best_line_ipas[i] = (
                            validation["source_ipas"][i],
                            validation["target_ipas"][i],
                        )
                        improved_lines.append((i + 1, old_score, score))

                if improved_lines:
                    improvements_str = ", ".join(
                        f"#{line} {old:.1%}→{new:.1%}"
                        for line, old, new in improved_lines
                    )
                    logger.info(f"   ✓ Improved: {improvements_str}")

                # Log current best state after each attempt
                avg_best = sum(best_line_scores) / len(best_line_scores)
                logger.info(f"   Current best average: {avg_best:.1%}")

                # Check if all lines pass threshold (based on best so far)
                all_pass = all(score >= self.config.similarity_threshold for score in best_line_scores)
                if all_pass:
                    logger.info(f"   ✓ All lines pass threshold on attempt {attempt}")
                    break

            except Exception as e:
                logger.warning(f"   Attempt {attempt} failed with error: {e}")
                if attempt == self.config.max_retries:
                    # On last attempt, if we have any best lines, use them; otherwise raise
                    if any(line is not None for line in best_lines):
                        logger.warning("   Using best lines from previous attempts")
                        break
                    raise
                # Otherwise continue to next retry
                user_prompt = f"""Previous attempt had errors. Please try again.

Original {source_lang} lyrics:
{source_lyrics}

Remember: Return ONLY valid JSON with "soramimi_lines" and "reasoning" fields.
Create {target_lang} soramimi that SOUNDS like the {source_lang}."""
                continue

            # If not last attempt, prepare feedback for retry based on BEST lines so far
            if attempt < self.config.max_retries:
                locked_lines = []  # Lines above threshold - MUST NOT CHANGE
                improve_lines = []  # Lines below threshold - need improvement

                for i, (best_score, best_line) in enumerate(
                    zip(best_line_scores, best_lines), 1
                ):
                    if best_line is None:
                        improve_lines.append(f"{i}. (need line)")
                    elif best_score >= self.config.similarity_threshold:
                        locked_lines.append(f"{i}. {best_line}")
                    else:
                        improve_lines.append(f"{i}. {best_line} ({best_score:.1%})")

                overall_best = sum(best_line_scores) / len(best_line_scores) if best_line_scores else 0
                logger.info(
                    f"   Attempt {attempt}: best overall {overall_best:.1%}"
                )

                # Build retry prompt - LOCK passing lines
                parts = [f"Attempt {attempt}. Target: {self.config.similarity_threshold:.0%}+"]

                if locked_lines:
                    parts.append("\nLOCKED (output these EXACTLY as shown):")
                    parts.extend(locked_lines)

                if improve_lines:
                    parts.append("\nIMPROVE (match sounds better):")
                    parts.extend(improve_lines)

                user_prompt = "\n".join(parts)
            else:
                if attempt == self.config.max_retries:
                    logger.info(f"   Max retries ({self.config.max_retries}) reached")

        # Build final result from best lines
        overall_similarity = sum(best_line_scores) / len(best_line_scores) if best_line_scores else 0
        all_pass = all(score >= self.config.similarity_threshold for score in best_line_scores)

        # Ensure all best lines were set (no None values)
        if any(line is None for line in best_lines):
            raise RuntimeError(
                f"Some lines were never successfully generated: "
                f"{[i+1 for i, line in enumerate(best_lines) if line is None]}"
            )

        # Create translation object from best lines
        # IMPORTANT: This uses best_lines which contains the highest-scoring line for each position
        translation = SoramimiTranslation(
            soramimi_lines=list(best_lines),  # Create new list to avoid reference issues
            source_ipa=[ipa[0] for ipa in best_line_ipas],
            target_ipa=[ipa[1] for ipa in best_line_ipas],
            similarity_scores=list(best_line_scores),  # Create new list
            overall_similarity=overall_similarity,
            reasoning=f"Combined best lines from {attempt} attempts",
        )

        # Verify the translation matches our best lines
        assert len(translation.soramimi_lines) == len(best_lines), "Line count mismatch"
        assert translation.soramimi_lines == best_lines, "Translation doesn't match best lines!"
        assert translation.similarity_scores == best_line_scores, "Scores don't match!"

        # Create validation dict for display
        validation = {
            "source_ipas": translation.source_ipa,
            "target_ipas": translation.target_ipa,
            "similarities": translation.similarity_scores,
            "overall_similarity": overall_similarity,
            "passed": all_pass,
            "feedback": "Best result per line",
        }

        logger.info(
            f"   Soramimi creation completed - using best per-line result: {overall_similarity:.1%}"
        )

        # Log final per-line scores for verification
        logger.info("   Final best lines:")
        for i, (line, score) in enumerate(zip(best_lines, best_line_scores), 1):
            status = "PASS" if score >= self.config.similarity_threshold else "FAIL"
            logger.info(f"      {i}. {line} ({score:.1%} {status})")

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
