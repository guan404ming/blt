"""
Lyrics Translator using PydanticAI + Gemini 2.0 Flash
Core translator implementation
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional
from pydantic_ai import Agent

from .models import LyricTranslation, MusicConstraints
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator

# Setup logger
logger = logging.getLogger(__name__)


class LyricsTranslator:
    """Lyrics translator with constraint validation"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        auto_save: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        Initialize translator

        Args:
            model: Gemini model name
            api_key: Google AI API Key (reads from env if not provided)
            auto_save: Auto-save translation results
            save_dir: Save directory (defaults to 'outputs')
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")

        # Set API key in environment for pydantic-ai
        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.auto_save = auto_save
        self.save_dir = save_dir or "outputs"

        # Toolbox - ConstraintValidator serves as both validator and toolbox
        self.feature_extractor = FeatureExtractor()
        self.validator = ConstraintValidator()

        # Tool call tracking
        self.tool_call_stats = defaultdict(int)

        # Initialize Agent - pydantic-ai will infer Google provider from model name
        self.agent = Agent(
            model=model,
            output_type=LyricTranslation,
            system_prompt=self._get_system_prompt(),
        )

        # Register tools from ConstraintValidator for LLM use
        self._register_tools_from_validator()

    def _register_tools_from_validator(self):
        """Register tools from ConstraintValidator for LLM to call"""

        # Wrap validator methods as concise tool functions with logging
        def verify_all_constraints(
            lines: list[str],
            language: str,
            target_syllables: list[int],
            rhyme_scheme: str = "",
        ) -> dict:
            """Verify all constraints at once (most efficient). Returns: {"syllables": [int], "syllables_match": bool, "rhyme_endings": [str], "rhymes_valid": bool, "feedback": str}"""
            self.tool_call_stats["verify_all_constraints"] += 1
            logger.info(
                f"🔧 Tool called: verify_all_constraints(lines={len(lines)}, language={language}, target_syllables={target_syllables})"
            )
            result = self.validator.verify_all_constraints(
                lines, language, target_syllables, rhyme_scheme
            )
            logger.info(
                f"   Result: syllables_match={result['syllables_match']}, rhymes_valid={result.get('rhymes_valid', 'N/A')}"
            )
            if result.get("feedback"):
                logger.info(f"   Feedback:\n{result['feedback']}")
            return result

        def count_syllables(text: str, language: str) -> int:
            """Count syllables using IPA-based method (phonemizer + espeak-ng). Converts text to IPA and counts vowel nuclei. Diphthongs and long vowels counted as single syllable. Use verify_all_constraints for multiple lines."""
            self.tool_call_stats["count_syllables"] += 1
            logger.info(
                f"🔧 Tool called: count_syllables(text='{text[:30]}...', language={language})"
            )
            result = self.validator.count_syllables(text, language)
            logger.info(f"   Result: {result} syllables")
            return result

        def check_rhyme(text1: str, text2: str, language: str) -> dict:
            """Check if two texts rhyme. Returns: {"rhymes": bool, "rhyme1": str, "rhyme2": str}"""
            self.tool_call_stats["check_rhyme"] += 1
            logger.info(
                f"🔧 Tool called: check_rhyme(text1='{text1[:20]}...', text2='{text2[:20]}...', language={language})"
            )
            result = self.validator.check_rhyme(text1, text2, language)
            logger.info(f"   Result: rhymes={result['rhymes']}")
            return result

        def get_syllable_pattern(text: str, language: str) -> dict:
            """Analyze text: segment into words (LLM-based) and count syllables for each word (IPA-based with phonemizer + espeak-ng). Returns: {"words": [str], "syllables": [int], "total_syllables": int}. Example: "I like tomato" → {"words": ["I", "like", "tomato"], "syllables": [1, 1, 3], "total_syllables": 5}"""
            self.tool_call_stats["get_syllable_pattern"] += 1
            logger.info(
                f"🔧 Tool called: get_syllable_pattern(text='{text[:30]}...', language={language})"
            )
            syllables = self.feature_extractor._get_syllable_pattern(text, language)
            result = {
                "syllables": syllables,
            }
            logger.info(
                f"   Result: {result['total_syllables']}"
            )
            logger.info(f"   Syllables: {syllables}")
            return result

        # Register tools to Agent
        self.agent.tool_plain(verify_all_constraints)
        self.agent.tool_plain(count_syllables)
        self.agent.tool_plain(check_rhyme)
        self.agent.tool_plain(get_syllable_pattern)

    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        return """You are a professional lyrics translation expert specialized in singable translations.

CONSTRAINT PRIORITIES (strictly enforced in this order):
1. SYLLABLE COUNT (CRITICAL) - Must match exactly
2. Rhyme scheme (IMPORTANT) - Match when possible, syllable count takes precedence
3. WORD COUNT (HELPFUL) - Try to match the number of words per line for better singability

AVAILABLE TOOLS (syllable counting uses IPA-based method with phonemizer + espeak-ng):
- verify_all_constraints(lines, language, target_syllables, rhyme_scheme) - Check syllable and rhyme constraints at once
- get_syllable_pattern(text, language) - Get syllables pattern (IPA-based). Returns {"syllables": [int]}
- count_syllables(text, language) - Count total syllables using IPA vowel nuclei detection
- check_rhyme(text1, text2, language) - Check if two texts rhyme using IPA rhyme ending comparison

WORKFLOW:
1. Draft all translations (prioritize syllable count over grammar perfection)
2. Call verify_all_constraints to check syllable and rhyme constraints
3. Read the 'feedback' field to see exactly which lines need adjustment and by how much
4. If syllables_match=False, adjust the specific lines mentioned in feedback
5. Use get_syllable_pattern to understand word-level syllable breakdown for fine-tuning
6. Re-verify until syllables_match=True, then output

Limit to 10 verification rounds. If still mismatched, output best attempt with reasoning."""

    def translate(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        constraints: Optional[MusicConstraints] = None,
        save_path: Optional[str] = None,
        save_format: str = "json",
    ) -> LyricTranslation:
        """
        Translate lyrics

        Args:
            source_lyrics: Source language lyrics
            source_lang: Source language
            target_lang: Target language
            constraints: Music constraints (auto-extracted if not provided)
            save_path: Save path (overrides auto_save setting)
            save_format: Save format ("json", "txt", "md")

        Returns:
            LyricTranslation: Translation result
        """
        start_time = time.time()

        # Reset tool call stats for this translation
        self.tool_call_stats.clear()

        # 1. Extract constraints (if not provided)
        if constraints is None:
            self.feature_extractor.source_lang = source_lang
            self.feature_extractor.target_lang = target_lang
            constraints = self.feature_extractor.extract_constraints(source_lyrics)

        # 2. Build prompt
        user_prompt = self._build_prompt(
            source_lyrics=source_lyrics,
            source_lang=source_lang,
            target_lang=target_lang,
            constraints=constraints,
        )

        # 3. Call LLM (only outputs translated_lines and reasoning)
        logger.info("🚀 Starting translation with LLM...")
        result = self.agent.run_sync(user_prompt)
        translation = result.output
        logger.info("✅ LLM translation completed")

        # 4. Calculate syllable counts, rhyme endings, and word segments using validator
        translation.syllable_counts = [
            self.validator.count_syllables(line, target_lang)
            for line in translation.translated_lines
        ]
        translation.rhyme_endings = [
            self.validator.extractor._extract_rhyme_ending(line, target_lang)
            for line in translation.translated_lines
        ]
        translation.syllable_patterns = [
            self.feature_extractor._get_syllable_pattern(line, target_lang)
            for line in translation.translated_lines
        ]

        # 5. Add tool call statistics to translation
        translation.tool_call_stats = dict(self.tool_call_stats)

        # 6. Validate and display result
        self.validator.target_lang = target_lang
        validation_result = self.validator.validate(translation, constraints)

        elapsed_time = time.time() - start_time

        if validation_result.passed:
            print(f"✓ All constraints satisfied (took {elapsed_time:.1f}s)")
        else:
            print(f"⚠ Score: {validation_result.score:.0%} (took {elapsed_time:.1f}s)")

        # Display tool call stats
        if self.tool_call_stats:
            print("\n📊 Tool Call Statistics:")
            for tool_name, count in sorted(self.tool_call_stats.items()):
                print(f"   {tool_name}: {count}")

        # 7. Save result (if enabled)
        if save_path or self.auto_save:
            self._save_translation(
                translation,
                save_path=save_path,
                save_format=save_format,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        return translation

    def _build_prompt(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        constraints: MusicConstraints,
    ) -> str:
        """Build translation prompt"""
        prompt_parts = [
            f"TRANSLATE FROM {source_lang} TO {target_lang}",
            "",
            "SOURCE LYRICS:",
            source_lyrics,
            "",
            "CONSTRAINTS:",
            f"• Syllable counts per line: {constraints.syllable_counts}",
        ]

        if constraints.rhyme_scheme:
            prompt_parts.append(f"• Rhyme scheme: {constraints.rhyme_scheme}")

        if constraints.syllable_patterns:
            word_counts = [len(pattern) for pattern in constraints.syllable_patterns]
            prompt_parts.append(f"• Word counts per line: {word_counts}")
            prompt_parts.append("• Source syllable patterns:")
            for i, pattern in enumerate(constraints.syllable_patterns, 1):
                prompt_parts.append(f"  Line {i}: [{', '.join(str(s) for s in pattern)}]")

        prompt_parts.append("")
        prompt_parts.append(
            "Translate ensuring all constraints are met. Use verify_all_constraints for verification and get_syllable_pattern for word-level analysis."
        )

        return "\n".join(prompt_parts)

    def _save_translation(
        self,
        translation: LyricTranslation,
        save_path: Optional[str] = None,
        save_format: str = "json",
        source_lang: str = "Unknown",
        target_lang: str = "Unknown",
    ) -> None:
        """
        Save translation result

        Args:
            translation: Translation result
            save_path: Save path (auto-generated if not provided)
            save_format: Save format
            source_lang: Source language
            target_lang: Target language
        """
        if save_path is None:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"translation_{source_lang}_to_{target_lang}_{timestamp}.{save_format}"
            )
            save_path = os.path.join(self.save_dir, filename)

        # Save
        translation.save(save_path, format=save_format)
        print(f"\n✓ Translation saved to: {save_path}")
