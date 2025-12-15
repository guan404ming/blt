"""
Prompt templates for translation workflows
"""

from langchain_core.messages import SystemMessage


def get_lyrics_translation_system_prompt(config) -> SystemMessage:
    """
    Get system prompt for lyrics translation

    Args:
        config: LyricsTranslationAgentConfig instance

    Returns:
        SystemMessage with translation instructions
    """
    system_prompt = config.get_system_prompt() if hasattr(config, 'get_system_prompt') else (
        """You are an expert lyrics translator. Your task is to translate lyrics while preserving:
1. Syllable count constraints
2. Rhyme schemes
3. Poetic meaning and emotion
4. Cultural context

For each line:
1. First, understand the source meaning
2. Find a translation that matches the syllable count
3. Use the count_syllables tool to verify each translation
4. If the count doesn't match, revise until it's correct
5. Ensure rhyme schemes are preserved when possible

Be precise with syllable counts - they are critical constraints."""
    )
    return SystemMessage(content=system_prompt)


def get_soramimi_system_prompt() -> SystemMessage:
    """
    Get system prompt for soramimi translation

    Returns:
        SystemMessage with soramimi instructions
    """
    system_prompt = """You are an expert in soramimi (空耳) translation - creating words that sound similar across languages.

Your task is to create target language text that sounds phonetically similar to the source text.

Process:
1. Analyze the source phonemes using get_phoneme_ipa
2. For each phoneme, find a target language character/syllable with similar sound
3. Build a complete phoneme mapping
4. Use check_phonetic_similarity to verify your mapping produces good phonetic matches
5. Refine based on similarity scores

Key principles:
- Focus on phonetic similarity, not meaning
- Preserve the rhythm and flow of the source
- Use natural characters/syllables from the target language
- Aim for high phonetic similarity (70%+ similarity score)"""
    return SystemMessage(content=system_prompt)


def get_lyrics_translation_user_prompt(source_lyrics: str, target_lang: str) -> str:
    """
    Get user prompt for lyrics translation

    Args:
        source_lyrics: The source lyrics to translate
        target_lang: Target language code

    Returns:
        User prompt string
    """
    return f"""Translate the following lyrics to {target_lang}, following all syllable constraints and rhyme schemes:

{source_lyrics}

Remember to:
1. Use the count_syllables tool to verify each translation's syllable count
2. Preserve the meaning and emotional impact
3. Maintain the original rhyme scheme when possible
4. Work line by line for accuracy"""


def get_soramimi_user_prompt(
    source_lyrics: str, source_lang: str, target_lang: str
) -> str:
    """
    Get user prompt for soramimi translation

    Args:
        source_lyrics: The source lyrics
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        User prompt string
    """
    return f"""Create a soramimi (空耳) translation from {source_lang} to {target_lang}.

Source lyrics:
{source_lyrics}

Use the tools to:
1. Extract phonemes from the source
2. Map each phoneme to similar-sounding {target_lang} characters
3. Verify phonetic similarity with check_phonetic_similarity
4. Build the complete soramimi text

Target {target_lang} characters should sound like the source {source_lang} phonemes."""
