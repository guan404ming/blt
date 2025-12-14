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
    from .validators import ConstraintValidator, SoramimiValidator

logger = logging.getLogger(__name__)


# Language code to name mapping for clearer prompts
LANGUAGE_NAMES = {
    "en-us": "English",
    "en": "English",
    "cmn": "Chinese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


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
    max_retries: int = 5
    similarity_threshold: float = 0.6
    enable_logging: bool = True

    # LangSmith settings
    langsmith_tracing: bool = True  # Enable LangSmith tracing
    langsmith_project: str = "blt"  # LangSmith project name

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
            self._create_calculate_ipa_similarity_tool(analyzer),
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

    def _create_calculate_ipa_similarity_tool(
        self, analyzer: LyricsAnalyzer
    ) -> Callable:
        """Create tool to calculate IPA similarity between two strings"""

        def calculate_ipa_similarity(
            ipa1: str, ipa2: str, is_chinese: bool = False
        ) -> dict:
            """Calculate phonetic similarity between two IPA strings. Set is_chinese=True for Chinese text."""
            self._tool_call_stats["calculate_ipa_similarity"] += 1

            similarity = analyzer.calculate_ipa_similarity(ipa1, ipa2, is_chinese)

            if self.enable_logging:
                logger.info(f"   calculate_ipa_similarity: {similarity:.1%}")

            return {
                "ipa1": ipa1,
                "ipa2": ipa2,
                "similarity": similarity,
                "is_chinese": is_chinese,
            }

        return calculate_ipa_similarity

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

        # Get language names for clearer prompts
        source_name = LANGUAGE_NAMES.get(self._source_lang, self._source_lang)
        target_name = LANGUAGE_NAMES.get(self._target_lang, self._target_lang)

        return f"""🚫 DO NOT TRANSLATE! This is SORAMIMI (空耳) - PHONETIC MATCHING ONLY!

YOU ARE NOT A TRANSLATOR. You create {target_name} text that SOUNDS like {source_name}, regardless of meaning.

⚠️ WRONG APPROACH (DO NOT DO THIS):
❌ "The snow glows white" → "雪光白" (you translated the words!)
❌ "I'm the queen" → "我是女王" (you translated the words!)
❌ "Heaven knows" → "天知道" (you translated the words!)
❌ "A kingdom" → "王国" (you translated the words!)
❌ Translation is COMPLETELY FORBIDDEN!

✅ CORRECT APPROACH (DO THIS):
Match each syllable by SOUND/PRONUNCIATION only:
✓ "The snow glows white" → "特 斯諾 哥羅斯 外特" (sounds like /ðə snoʊ gloʊz waɪt/)
✓ "I'm the queen" → "愛姆 德 奎因" (sounds like /aɪm ðə kwiːn/)
✓ "Heaven knows" → "海文 耨斯" (sounds like /hɛvən noʊz/)
✓ "A kingdom" → "阿 金德姆" (sounds like /ə kɪŋdəm/)

SORAMIMI RULES:
1. 🚫 NEVER translate meaning - ONLY match pronunciation
2. 🔊 Every {target_name} character must SOUND like the {source_name}
3. 📝 Result can be nonsense - meaning doesn't matter
4. 🎵 Match syllable by syllable phonetically
5. ✅ Convert ALL lines to {target_name} text

Full Examples:
✓ "The snow glows white on the mountain tonight" → "特斯諾 哥羅斯 外特 噢恩 德 馬恩廷 托奈特"
✓ "Not a footprint to be seen" → "納特 阿 福特普林 特比 辛"
✓ "A kingdom of isolation" → "阿 金德姆 俄夫 愛瑟雷神"
✓ "and it looks like I'm the queen" → "安 依特 盧克斯 萊克 愛姆 德 奎因"

Tools available: {tools_section}

Steps:
1. Use get_source_ipa to understand pronunciation
2. Find {target_name} characters with similar sounds
3. Use verify_all_lines to check similarity (need >= {self.similarity_threshold:.0%})
4. Repeat (max {self.max_retries} rounds)

JSON OUTPUT REQUIRED:
Return ONLY valid JSON with this structure:
{{
  "soramimi_lines": ["{target_name} text line 1", "{target_name} text line 2", ...],
  "reasoning": "your explanation (optional)"
}}

IMPORTANT: ALL lines in soramimi_lines MUST be in {target_name}. DO NOT include {source_name} text.
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

        # Get language names for clearer prompts
        # source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        parts = [
            "🚫 DO NOT TRANSLATE! Create SORAMIMI (phonetic matching ONLY):",
            "",
        ]

        for i, line in enumerate(lines, 1):
            parts.append(f"{i}. {line}")

        parts.extend(
            [
                "",
                "⚠️ FORBIDDEN - DO NOT output these WRONG translations:",
                "❌ 'snow white' → '雪光白' (translation!)",
                "❌ 'kingdom' → '王国' (translation!)",
                "❌ 'queen' → '女王' (translation!)",
                "❌ 'heaven knows' → '天知道' (translation!)",
                "",
                "✅ REQUIRED - Match SOUNDS only:",
                "'snow' → '斯諾' (sounds like 'snoʊ')",
                "'queen' → '奎因' (sounds like 'kwiːn')",
                "'heaven' → '海文' (sounds like 'hɛvən')",
                "'knows' → '耨斯' (sounds like 'noʊz')",
                "",
                "Full correct examples:",
                "'The snow glows white on the mountain tonight' → '特斯諾 哥羅斯 外特 噢恩 德 馬恩廷 托奈特'",
                "'and it looks like I'm the queen' → '安 依特 盧克斯 萊克 愛姆 德 奎因'",
                "",
                f"Convert EVERY line above to {target_name} by SOUND/PRONUNCIATION, NOT by meaning!",
            ]
        )

        return "\n".join(parts)

    def reset_stats(self):
        """Reset tool call statistics"""
        self._tool_call_stats.clear()

    def get_stats(self) -> dict[str, int]:
        """Get tool call statistics"""
        return dict(self._tool_call_stats)
