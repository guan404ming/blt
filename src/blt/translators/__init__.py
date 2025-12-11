"""Lyrics Translation module.

This module provides:
- Gemini-based lyrics translation with constraint validation
- Music constraints extraction and validation
- Support for multiple languages with syllable/rhyme preservation
- Modular architecture with unified analyzer

Architecture:
    LyricsAnalyzer: Core analysis (syllables, rhymes, patterns)
    ConstraintValidator: Validation logic
    TranslatorConfig: Configuration + prompts + tool registration
    LyricsTranslator: Main orchestrator
"""

# Core components
from .analyzer import LyricsAnalyzer
from .validator import ConstraintValidator

# Configuration (includes prompts + tools)
from .config import TranslatorConfig

# Models
from .models import LyricTranslation, MusicConstraints, ValidationResult

# Main translator
from .translator import LyricsTranslator

__version__ = "0.1.0"

__all__ = [
    # Main
    "LyricsTranslator",
    # Core
    "LyricsAnalyzer",
    "ConstraintValidator",
    "TranslatorConfig",
    # Models
    "LyricTranslation",
    "MusicConstraints",
    "ValidationResult",
]
