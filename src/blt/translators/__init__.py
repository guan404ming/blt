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
    TranslatorConfig: Configuration + prompts + tool registration
    SoramimiConfig: Configuration for soramimi translation
    LyricsTranslationAgent: Main orchestrator for constraint-based translation
    SoramimiTranslationAgent: Main orchestrator for phonetic translation
"""

# Core components
from .analyzer import LyricsAnalyzer
from .validator import ConstraintValidator, SoramimiValidator

# Configuration (includes prompts + tools)
from .config import TranslatorConfig, SoramimiConfig

# Models
from .models import (
    LyricTranslation,
    MusicConstraints,
    ValidationResult,
    SoramimiTranslation,
)

# Translators
from .agents import LyricsTranslationAgent, SoramimiTranslationAgent

__version__ = "0.1.0"

__all__ = [
    # Main agents
    "LyricsTranslationAgent",
    "SoramimiTranslationAgent",
    # Core
    "LyricsAnalyzer",
    "ConstraintValidator",
    "SoramimiValidator",
    "TranslatorConfig",
    "SoramimiConfig",
    # Models
    "LyricTranslation",
    "MusicConstraints",
    "ValidationResult",
    "SoramimiTranslation",
]
