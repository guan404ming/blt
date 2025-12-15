"""Lyrics Translation module.

This module provides:
- Ollama-based local LLM lyrics translation with constraint validation
- Music constraints extraction and validation
- Support for multiple languages with syllable/rhyme preservation
- Soramimi (phonetic) translation for sound-alike lyrics
- Modular architecture with unified analyzer

Architecture:
    LyricsAnalyzer: Core analysis (syllables, rhymes, patterns, IPA)
    ConstraintValidator: Validation logic for constraint-based translation
    SoramimiValidator: Validation logic for phonetic similarity
    LyricsTranslationAgentConfig: Configuration + prompts + tool registration
    SoramimiTranslationAgentConfig: Configuration for soramimi translation
    LyricsTranslationAgent: Main orchestrator for constraint-based translation
    SoramimiTranslationAgent: Main orchestrator for phonetic translation
"""

# Core components
from .shared import (
    LyricsAnalyzer,
    LyricTranslation,
    MusicConstraints,
    ValidationResult,
    SoramimiTranslation,
)

# Agent configurations
from .lyrics_translation.config import LyricsTranslationAgentConfig
from .soramimi_translation.config import SoramimiTranslationAgentConfig

# Translators
from .lyrics_translation import LyricsTranslationAgent
from .soramimi_translation import SoramimiTranslationAgent

__version__ = "0.1.0"

__all__ = [
    # Main agents
    "LyricsTranslationAgent",
    "SoramimiTranslationAgent",
    # Core
    "LyricsAnalyzer",
    "LyricsTranslationAgentConfig",
    "SoramimiTranslationAgentConfig",
    # Models
    "LyricTranslation",
    "MusicConstraints",
    "ValidationResult",
    "SoramimiTranslation",
]
