"""
State definitions for LangGraph workflows
"""

from typing import TypedDict, Optional, Annotated
from operator import add


class LyricsTranslationState(TypedDict):
    """State for constraint-based lyrics translation graph"""

    # Input
    source_lyrics: str
    source_lang: str
    target_lang: str

    # Constraints
    constraints: Optional[dict]  # MusicConstraints as dict
    syllable_counts: Optional[list[int]]
    rhyme_scheme: Optional[str]
    syllable_patterns: Optional[list[str]]

    # Translation
    translated_lines: Optional[list[str]]
    reasoning: Optional[str]

    # Metrics
    translation_syllable_counts: Optional[list[int]]
    translation_rhyme_endings: Optional[list[str]]
    translation_syllable_patterns: Optional[list[str]]

    # Validation
    validation_passed: Optional[bool]
    validation_score: Optional[float]

    # Control
    attempt: int
    max_attempts: int
    all_lines_done: Optional[bool]
    messages: Annotated[list, add]


class SoramimiMappingState(TypedDict):
    """State for mapping-based soramimi translation graph"""

    # Source information
    source_lines: list[str]
    source_lang: str
    target_lang: str

    # Phoneme mapping
    source_phonemes: list[str]  # Unique phonemes from source
    phoneme_mapping: dict[str, str]  # phoneme -> target character/syllable
    mapping_scores: dict[str, float]  # phoneme -> similarity score

    # Current translation
    soramimi_lines: Optional[list[str]]
    source_ipa: Optional[list[str]]
    target_ipa: Optional[list[str]]
    similarity_scores: Optional[list[float]]
    overall_similarity: Optional[float]

    # Best results
    best_mapping: Optional[dict[str, str]]
    best_lines: Optional[list[str]]
    best_scores: Optional[list[float]]
    best_ipas: Optional[list[tuple[str, str]]]

    # Control
    attempt: int
    max_attempts: int
    threshold: float
    messages: Annotated[list, add]


def create_lyrics_translation_initial_state(
    source_lyrics: str,
    source_lang: str,
    target_lang: str,
    constraints,
) -> LyricsTranslationState:
    """
    Create initial state for lyrics translation graph

    Args:
        source_lyrics: Source lyrics text
        source_lang: Source language code
        target_lang: Target language code
        constraints: MusicConstraints object with syllable_counts, rhyme_scheme, syllable_patterns

    Returns:
        LyricsTranslationState initialized with all required fields
    """
    return {
        "source_lyrics": source_lyrics,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "constraints": None,  # Not used directly in graph
        "syllable_counts": constraints.syllable_counts,
        "rhyme_scheme": constraints.rhyme_scheme,
        "syllable_patterns": constraints.syllable_patterns,
        "translated_lines": None,
        "reasoning": None,
        "translation_syllable_counts": None,
        "translation_rhyme_endings": None,
        "translation_syllable_patterns": None,
        "validation_passed": None,
        "validation_score": None,
        "attempt": 1,
        "max_attempts": 3,
        "all_lines_done": None,
        "messages": [],
    }


def create_soramimi_mapping_initial_state(
    source_lines: list[str],
    source_lang: str,
    target_lang: str,
    max_attempts: int,
    threshold: float,
) -> SoramimiMappingState:
    """
    Create initial state for soramimi mapping graph

    Args:
        source_lines: List of source lyrics lines
        source_lang: Source language code
        target_lang: Target language code
        max_attempts: Maximum number of mapping refinement attempts
        threshold: Similarity threshold for stopping refinement

    Returns:
        SoramimiMappingState initialized with all required fields
    """
    return {
        "source_lines": source_lines,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_phonemes": [],
        "phoneme_mapping": {},
        "mapping_scores": {},
        "soramimi_lines": None,
        "source_ipa": None,
        "target_ipa": None,
        "similarity_scores": None,
        "overall_similarity": None,
        "best_mapping": None,
        "best_lines": None,
        "best_scores": None,
        "best_ipas": None,
        "attempt": 1,
        "max_attempts": max_attempts,
        "threshold": threshold,
        "messages": [],
    }
