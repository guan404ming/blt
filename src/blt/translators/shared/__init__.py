"""Shared utilities for translation framework"""

from .analyzer import LyricsAnalyzer
from .tools import (
    count_syllables,
    text_to_ipa,
    extract_rhyme_ending,
    check_rhyme,
    get_syllable_patterns,
    detect_rhyme_scheme,
    calculate_ipa_similarity,
)

__all__ = [
    # Core analysis
    "LyricsAnalyzer",
    # Analyzer tools
    "count_syllables",
    "text_to_ipa",
    "extract_rhyme_ending",
    "check_rhyme",
    "get_syllable_patterns",
    "detect_rhyme_scheme",
    "calculate_ipa_similarity",
]
