"""
Tool definitions for lyrics analysis - wraps all analyzer functions
"""

from langchain_core.tools import tool
from .analyzer import LyricsAnalyzer

# Initialize analyzer instance for all tools
_analyzer = LyricsAnalyzer()


# ==================== ANALYZER TOOLS ====================


@tool
def count_syllables(text: str, language: str) -> int:
    """
    Count the number of syllables in the given text.

    Args:
        text: The text to count syllables in
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        The number of syllables in the text
    """
    return _analyzer.count_syllables(text, language)


@tool
def text_to_ipa(text: str, language: str) -> str:
    """
    Convert text to IPA (International Phonetic Alphabet) representation.

    Args:
        text: The text to convert
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        The IPA transcription
    """
    return _analyzer.text_to_ipa(text, language)


@tool
def extract_rhyme_ending(text: str, language: str) -> str:
    """
    Extract the rhyme ending (final syllable/sound) from text.

    Args:
        text: The text to analyze
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        The rhyme ending string
    """
    return _analyzer.extract_rhyme_ending(text, language)


@tool
def check_rhyme(text1: str, text2: str, language: str) -> bool:
    """
    Check if two pieces of text rhyme with each other.

    Args:
        text1: First text
        text2: Second text
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        True if texts rhyme, False otherwise
    """
    return _analyzer.check_rhyme(text1, text2, language)


@tool
def get_syllable_patterns(lines: list[str], language: str) -> list[list[int]]:
    """
    Get syllable pattern (syllables per word) for multiple lines.

    Args:
        lines: List of text lines to analyze
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        List of syllable patterns, e.g., [[1, 1, 3], [1, 2, 1]]
    """
    return _analyzer.get_syllable_patterns(lines, language)


@tool
def detect_rhyme_scheme(lines: list[str], language: str) -> str:
    """
    Detect the rhyme scheme from a list of lines.

    Args:
        lines: List of text lines
        language: The language code (e.g., 'en-us', 'cmn', 'ja')

    Returns:
        Rhyme scheme string (e.g., "AABB", "ABAB")
    """
    return _analyzer.detect_rhyme_scheme(lines, language)


@tool
def calculate_ipa_similarity(
    ipa1: str, ipa2: str, is_chinese: bool = False
) -> float:
    """
    Calculate phonetic similarity between two IPA strings.

    Args:
        ipa1: First IPA string (or Chinese text if is_chinese=True)
        ipa2: Second IPA string (or Chinese text if is_chinese=True)
        is_chinese: If True, convert Chinese to pinyin first

    Returns:
        Similarity score between 0 and 1
    """
    return _analyzer.calculate_ipa_similarity(ipa1, ipa2, is_chinese)
