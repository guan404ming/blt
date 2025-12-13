"""
Unified Configuration - Includes prompts and tool registration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING
from collections import defaultdict
from pydantic_ai import Agent

from .analyzer import LyricsAnalyzer

if TYPE_CHECKING:
    from .validator import ConstraintValidator, SoramimiValidator

logger = logging.getLogger(__name__)


@dataclass
class LyricsTranslationAgentConfig:
    """Unified configuration with prompts and tools"""

    # Model settings
    model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"  # Ollama model name
    ollama_base_url: str = "http://localhost:11434/v1"

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

JSON OUTPUT REQUIRED:
Return ONLY valid JSON with this structure:
{{
    translated_lines: list[str] = Field(description="Translated lyrics line by line")
    syllable_counts: list[int] = Field(
        description="Syllable count per line (LLM outputs, we recalculate)"
    )
    rhyme_endings: list[str] = Field(
        description="Rhyme ending per line (LLM outputs, we recalculate)"
    )
    syllable_patterns: Optional[list[list[int]]] = Field(
        default=None,
        description="Syllable patterns per line (LLM outputs, we recalculate)",
    )
    reasoning: str = Field(description="Translation reasoning and considerations")


}}

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


# ==================== SORAMIMI CONFIG ====================


@dataclass
class SoramimiTranslationAgentConfig:
    """Configuration for Soramimi Translator"""

    # Model settings
    model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"  # Ollama model name
    ollama_base_url: str = "http://localhost:11434/v1"

    # Output settings
    auto_save: bool = False
    save_dir: str = "outputs"
    save_format: str = "json"

    # Translation settings
    max_retries: int = 8  # Reduced from 20 - more is slower without much benefit
    similarity_threshold: float = 0.6
    enable_logging: bool = True

    # Language defaults
    default_source_lang: str = "en-us"
    default_target_lang: str = "cmn"

    # Internal
    _tool_call_stats: dict = field(default_factory=lambda: defaultdict(int))
    _registered_tools: list = field(default_factory=list)

    # Store source info for tools
    _source_lines: list = field(default_factory=list)
    _source_lang: str = ""
    _target_lang: str = ""

    def register_tools(
        self,
        agent: Agent,
        analyzer: LyricsAnalyzer,
        validator: "SoramimiValidator",
    ):
        """Register LLM tools for soramimi translation"""
        self._registered_tools.clear()

        tools = [
            self._create_get_source_ipa_tool(analyzer),
            self._create_check_ipa_similarity_tool(validator),
            self._create_verify_all_lines_tool(validator),
        ]

        for tool_func in tools:
            agent.tool_plain(tool_func)
            self._registered_tools.append(
                {
                    "name": tool_func.__name__,
                    "doc": tool_func.__doc__ or "",
                }
            )

    def _create_get_source_ipa_tool(self, analyzer: LyricsAnalyzer) -> Callable:
        """Create tool to get source IPA"""

        def get_source_ipa(line_number: int) -> dict:
            """Get IPA transcription of a source line. Line numbers start at 1."""
            self._tool_call_stats["get_source_ipa"] += 1

            if line_number < 1 or line_number > len(self._source_lines):
                return {
                    "error": f"Invalid line number. Valid: 1-{len(self._source_lines)}"
                }

            text = self._source_lines[line_number - 1]
            ipa = analyzer.text_to_ipa(text, self._source_lang)

            if self.enable_logging:
                logger.info(f"   get_source_ipa(line {line_number}): {ipa}")

            return {"line_number": line_number, "text": text, "ipa": ipa}

        return get_source_ipa

    def _create_check_ipa_similarity_tool(
        self, validator: "SoramimiValidator"
    ) -> Callable:
        """Create tool to check IPA similarity for a single line"""

        def check_ipa_similarity(
            line_number: int,
            target_text: str,
        ) -> dict:
            """Check IPA similarity between source line and your translation. Line numbers start at 1."""
            self._tool_call_stats["check_ipa_similarity"] += 1

            if line_number < 1 or line_number > len(self._source_lines):
                return {
                    "error": f"Invalid line number. Valid: 1-{len(self._source_lines)}"
                }

            source_text = self._source_lines[line_number - 1]
            result = validator.validate_single_line(
                source_text, target_text, self._source_lang, self._target_lang
            )

            if self.enable_logging:
                logger.info(
                    f"   check_ipa_similarity(line {line_number}): "
                    f"{result['similarity']:.1%} - {'PASS' if result['passed'] else 'FAIL'}"
                )

            return {
                "line_number": line_number,
                "source_text": source_text,
                "target_text": target_text,
                **result,
            }

        return check_ipa_similarity

    def _create_verify_all_lines_tool(self, validator: "SoramimiValidator") -> Callable:
        """Create tool to verify all lines at once"""

        def verify_all_lines(target_lines: list[str]) -> dict:
            """Verify IPA similarity for all translated lines at once. Most efficient - use this!"""
            self._tool_call_stats["verify_all_lines"] += 1

            if len(target_lines) != len(self._source_lines):
                return {
                    "error": f"Expected {len(self._source_lines)} lines, got {len(target_lines)}"
                }

            result = validator.compare_ipa(
                self._source_lines,
                target_lines,
                self._source_lang,
                self._target_lang,
            )

            if self.enable_logging:
                logger.info(
                    f"   verify_all_lines: {result['overall_similarity']:.1%} overall - "
                    f"{'PASS' if result['passed'] else 'FAIL'}"
                )

            return result

        return verify_all_lines

    def get_system_prompt(self) -> str:
        """Generate system prompt for soramimi translation"""
        tool_descriptions = []
        for tool in self._registered_tools:
            tool_descriptions.append(f"- {tool['name']}: {tool['doc']}")

        tools_section = (
            "\n".join(tool_descriptions)
            if tool_descriptions
            else "Tools will be available"
        )

        return f"""You create SORAMIMI (空耳): {self._target_lang} text that SOUNDS like {self._source_lang}.

空耳（日語：そらみみ，Soramimi）是一種將聽到的聲音（通常是外語歌曲歌詞）故意「幻聽」或「誤聽」
改寫成發音近似，以達到方便記憶的二次創作行為
其精髓是「音準即可」，即改寫後的句子在發音上相似，可以完全不成句。

Examples:
1. The snow glows white on the mountain tonight → 特斯諾 哥羅斯 外特 噢恩 德 馬恩廷 托奈特
2. Not a footprint to be seen → 納特 阿 福特普林 特比 辛
3. A kingdom of isolation → 阿 金德姆 俄夫 愛瑟雷神
4. and it looks like I'm the queen → 安 依特 盧克斯 萊克 愛姆 德 奎因

Tools available: {tools_section}

Steps:
1. Use get_source_ipa to understand pronunciation
2. Find {self._target_lang} characters with similar sounds
3. Use verify_all_lines to check similarity (need >= {self.similarity_threshold:.0%})
4. Repeat (max {self.max_retries} rounds)

JSON OUTPUT REQUIRED:
Return ONLY valid JSON with this structure:
{{
  "soramimi_lines": ["Chinese characters line 1", "Chinese characters line 2", ...],
  "reasoning": "your explanation (optional)"
}}
"""

    def get_user_prompt(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Generate user prompt"""
        lines = [
            line.strip() for line in source_lyrics.strip().split("\n") if line.strip()
        ]

        parts = [
            f"Create SORAMIMI: find {target_lang} characters that SOUND like this {source_lang}:",
            "",
        ]

        for i, line in enumerate(lines, 1):
            parts.append(f"{i}. {line}")

        parts.extend(
            [
                """空耳（日語：そらみみ，Soramimi）是一種將聽到的聲音（通常是外語歌曲歌詞）故意「幻聽」或「誤聽」
                改寫成發音近似，以達到趣味或方便記憶的二次創作行為
                其精髓是「音準即可」，即改寫後的句子在發音上相似，可以完全不成句。""",
                "有一些沒辦法拼出來的字可以用英文代替",
                "The snow glows white on the mountain tonight → 死鬧勾斯歪動某頓頭奈",
                "Not a footprint to be seen → 哪惹浮噴 to 比辛",
                "A kingdom of isolation → 欸 King 痘媽佛愛收雷神",
            ]
        )

        return "\n".join(parts)

    def reset_stats(self):
        """Reset tool call statistics"""
        self._tool_call_stats.clear()

    def get_stats(self) -> dict[str, int]:
        """Get tool call statistics"""
        return dict(self._tool_call_stats)
