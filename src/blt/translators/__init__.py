"""Lyrics Translation module.

This module provides:
- Ollama-based local LLM lyrics translation with constraint validation
- Music constraints extraction and validation
- Support for multiple languages with syllable/rhyme preservation
- Soramimi (phonetic) translation for sound-alike lyrics
- Gemini API-based simple translation
- Modular architecture with unified analyzer

Architecture:
    LyricsAnalyzer: Core analysis (syllables, rhymes, patterns, IPA)
    lyrics_translation.Validator: Validation logic for constraint-based translation
    soramimi_translation.Validator: Validation logic for phonetic similarity
    LyricsTranslationAgentConfig: Configuration + prompts + tool registration
    SoramimiTranslationAgentConfig: Configuration for soramimi translation
    LyricsTranslationAgent: Main orchestrator for constraint-based translation
    SoramimiTranslationAgent: Main orchestrator for phonetic translation
    GeminiTranslator: Simple Gemini API-based translator
"""

# Core components
from .shared import LyricsAnalyzer
from .lyrics_translator.models import (
    LyricTranslation,
    MusicConstraints,
)
from .soramimi_translator.models import SoramimiTranslation
from .gemini_translator.models import GeminiTranslation

# Agent configurations
from .lyrics_translator.config import LyricsTranslationAgentConfig
from .soramimi_translator.config import SoramimiTranslationAgentConfig
from .gemini_translator.config import GeminiTranslatorConfig

# Translators
from .lyrics_translator import LyricsTranslationAgent
from .soramimi_translator import SoramimiTranslationAgent
from .gemini_translator import GeminiTranslator, GeminiTranslationAgent

__version__ = "0.1.0"

__all__ = [
    # Main agents and translators
    "LyricsTranslationAgent",
    "SoramimiTranslationAgent",
    "GeminiTranslationAgent",
    "GeminiTranslator",
    # Core
    "LyricsAnalyzer",
    "LyricsTranslationAgentConfig",
    "SoramimiTranslationAgentConfig",
    "GeminiTranslatorConfig",
    # Models
    "LyricTranslation",
    "MusicConstraints",
    "SoramimiTranslation",
    "GeminiTranslation",
]
