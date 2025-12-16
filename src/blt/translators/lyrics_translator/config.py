"""Configuration for Lyrics Translation Agent"""

from __future__ import annotations

from dataclasses import dataclass


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
    """Configuration for Lyrics Translation Agent"""

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
