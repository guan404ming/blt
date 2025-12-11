"""
Lyrics Translator
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional
from pydantic_ai import Agent

from .models import LyricTranslation, MusicConstraints
from .analyzer import LyricsAnalyzer
from .validator import ConstraintValidator
from .config import TranslatorConfig

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

        # Validate API key
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = api_key

        # Core components
        self.analyzer = analyzer or LyricsAnalyzer()
        self.validator = ConstraintValidator(self.analyzer)

        # Initialize agent (without system prompt yet)
        self.agent = Agent(
            model=self.config.model,
            output_type=LyricTranslation,
            system_prompt="",  # Will be set dynamically
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
