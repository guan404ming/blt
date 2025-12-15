"""
Unified Configuration - Includes prompts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    """Unified configuration with prompts"""

    # Model settings
    model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"  # Ollama model name
    ollama_base_url: str = "http://localhost:11434/v1"

    # Output settings
    auto_save: bool = False
    save_dir: str = "outputs"
    save_format: str = "json"

    # Translation settings
    max_retries: int = 10
    enable_logging: bool = True

    # LangSmith settings
    langsmith_tracing: bool = True  # Enable LangSmith tracing
    langsmith_project: str = "blt"  # LangSmith project name

    # Language defaults
    default_source_lang: str = "en-us"
    default_target_lang: str = "cmn"

    # ==================== PROMPT GENERATION ====================

    def get_system_prompt(self) -> str:
        """Generate system prompt for lyrics translation"""
        return """You are a lyrics translator. Translate ONE line at a time while matching the target syllable count.

You have tools available:
- count_syllables: Check how many syllables your translation has
- verify_line: Verify if your translation matches the target

WORKFLOW:
1. Translate the source line
2. Use count_syllables to check your translation
3. If syllable count doesn't match, revise and check again
4. Once correct, output the final translation which contains no any punctuations

IMPORTANT: Always verify with tools before finalizing your answer."""

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
        lines = [_.strip() for _ in source_lyrics.strip().split("\n") if _.strip()]

        parts = [f"Translate these {len(lines)} lines to {target_lang}:", ""]

        # Show each line with its required syllable count
        for i, (line, count, syllable_pattern) in enumerate(
            zip(lines, syllable_counts, syllable_patterns), 1
        ):
            parts.append(
                f'Line {i} ({count} syllables required): "{line}", syllable_pattern: {syllable_pattern}'
            )

        parts.append("")
        parts.append("REMEMBER: Each Chinese character = 1 syllable. Count carefully!")

        return "\n".join(parts)


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

    def get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Generate system prompt for soramimi translation"""
        # Get language names for clearer prompts
        source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

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

Steps:
1. Understand pronunciation of the source text
2. Find {target_name} characters with similar sounds
3. Ensure similarity is >= {self.similarity_threshold:.0%}
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
