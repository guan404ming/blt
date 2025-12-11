"""
Unified Configuration - Includes prompts and tool registration
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict
from pydantic_ai import Agent

from .analyzer import LyricsAnalyzer
from .validator import ConstraintValidator

logger = logging.getLogger(__name__)


@dataclass
class TranslatorConfig:
    """Unified configuration with prompts and tools"""

    # Model settings
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None

    # Output settings
    auto_save: bool = False
    save_dir: str = "outputs"
    save_format: str = "json"

    # Translation settings
    max_retries: int = 15
    enable_logging: bool = True

    # Language defaults
    default_source_lang: str = "en-us"
    default_target_lang: str = "cmn"

    # Internal
    _tool_call_stats: dict = field(default_factory=lambda: defaultdict(int))
    _registered_tools: list = field(default_factory=list)

    # ==================== TOOL REGISTRATION ====================

    def register_tools(
        self, agent: Agent, analyzer: LyricsAnalyzer, validator: ConstraintValidator
    ):
        """Register all LLM tools and collect their docstrings"""
        self._registered_tools.clear()

        # Create tools
        tools = [
            self._create_verify_all_tool(validator),
            self._create_count_syllables_tool(analyzer),
            self._create_check_rhyme_tool(analyzer),
            self._create_get_pattern_tool(analyzer),
        ]

        # Register and store metadata
        for tool_func in tools:
            agent.tool_plain(tool_func)
            self._registered_tools.append(
                {
                    "name": tool_func.__name__,
                    "doc": tool_func.__doc__ or "",
                }
            )

    def _create_verify_all_tool(self, validator: ConstraintValidator) -> Callable:
        """Create verify_all_constraints tool"""

        def verify_all_constraints(
            lines: list[str],
            language: str,
            target_syllables: list[int],
            rhyme_scheme: str = "",
            target_patterns: list[list[int]] | None = None,
        ) -> dict:
            """Check all constraints at once. Most efficient - use this first!"""
            self._tool_call_stats["verify_all_constraints"] += 1

            result = validator.verify_all_constraints(
                lines, language, target_syllables, rhyme_scheme, target_patterns
            )

            if self.enable_logging:
                logger.info(
                    f"🔧 verify_all_constraints: {len(lines)} lines, "
                    f"match={result['syllables_match']}"
                )
                if result.get("feedback") and not result["syllables_match"]:
                    logger.info(f"   {result['feedback']}")

            return result

        return verify_all_constraints

    def _create_count_syllables_tool(self, analyzer: LyricsAnalyzer) -> Callable:
        """Create count_syllables tool"""

        def count_syllables(text: str, language: str) -> int:
            """Count syllables in text using IPA method."""
            self._tool_call_stats["count_syllables"] += 1
            result = analyzer.count_syllables(text, language)

            if self.enable_logging:
                logger.info(f"🔧 count_syllables: {result}")

            return result

        return count_syllables

    def _create_check_rhyme_tool(self, analyzer: LyricsAnalyzer) -> Callable:
        """Create check_rhyme tool"""

        def check_rhyme(text1: str, text2: str, language: str) -> dict:
            """Check if two texts rhyme."""
            self._tool_call_stats["check_rhyme"] += 1

            rhyme1 = analyzer.extract_rhyme_ending(text1, language)
            rhyme2 = analyzer.extract_rhyme_ending(text2, language)
            rhymes = analyzer.check_rhyme(text1, text2, language)

            if self.enable_logging:
                logger.info(f"🔧 check_rhyme: {rhymes}")

            return {"rhymes": rhymes, "rhyme1": rhyme1, "rhyme2": rhyme2}

        return check_rhyme

    def _create_get_pattern_tool(self, analyzer: LyricsAnalyzer) -> Callable:
        """Create get_syllable_pattern tool"""

        def get_syllable_pattern(lines: list[str], language: str) -> dict:
            """Get word-level syllable breakdown for lines."""
            self._tool_call_stats["get_syllable_pattern"] += 1
            patterns = analyzer.get_syllable_patterns(lines, language)

            if self.enable_logging:
                logger.info(f"🔧 get_syllable_pattern: {len(lines)} lines")

            return {"syllable_patterns": patterns}

        return get_syllable_pattern

    # ==================== PROMPT GENERATION ====================

    def get_system_prompt(self) -> str:
        """Generate system prompt dynamically from registered tools"""
        # Build tool descriptions from docstrings
        tool_descriptions = []
        for tool in self._registered_tools:
            tool_descriptions.append(f"- {tool['name']}: {tool['doc']}")

        tools_section = (
            "\n".join(tool_descriptions)
            if tool_descriptions
            else "Tools will be available"
        )

        return f"""You are a lyrics translator. Preserve constraints in this priority order:

CRITICAL CONSTRAINTS (MUST MATCH):
1. Syllable patterns - The word-level syllable structure (e.g., [2, 2, 1, 2]) MUST match exactly
2. Total syllable count - Each line's total syllables MUST match the target

OPTIONAL CONSTRAINTS (nice to have):
3. Rhyme scheme - Try to preserve rhymes but this is the lowest priority

IMPORTANT: Syllable patterns are MORE important than total count. If you must choose,
prioritize matching the pattern structure over just hitting the syllable total.

Tools:
{tools_section}

Workflow:
1. Draft translation focusing on matching syllable patterns FIRST
2. Call verify_all_constraints to check all constraints
3. Read feedback carefully - fix pattern mismatches as top priority
4. If patterns don't match, restructure the line completely
5. Repeat (max {self.max_retries} rounds)

Output final translation."""

    def get_user_prompt(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        syllable_counts: list[int],
        rhyme_scheme: str = "",
        syllable_patterns: list[list[int]] = None,
    ) -> str:
        """Generate user prompt"""
        parts = [
            f"Translate from {source_lang} to {target_lang}:",
            "",
            source_lyrics,
            "",
            f"Syllables: {syllable_counts}",
        ]

        if rhyme_scheme:
            parts.append(f"Rhyme: {rhyme_scheme}")

        if syllable_patterns:
            parts.append("")
            parts.append("Patterns:")
            for i, p in enumerate(syllable_patterns, 1):
                parts.append(f"  {i}. {p}")

        return "\n".join(parts)

    # ==================== STATS ====================

    def reset_stats(self):
        """Reset tool call statistics"""
        self._tool_call_stats.clear()

    def get_stats(self) -> dict[str, int]:
        """Get tool call statistics"""
        return dict(self._tool_call_stats)
