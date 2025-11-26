"""AI-Powered Lyrics Translation module.

This module provides:
- PydanticAI + Gemini-based lyrics translation
- Music constraints extraction and validation
- Support for multiple languages with syllable/rhyme preservation
"""

from .translator import LyricsTranslator
from .models import LyricTranslation, MusicConstraints, ValidationResult
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator

__version__ = "0.2.0"

__all__ = [
    "LyricsTranslator",
    "LyricTranslation",
    "MusicConstraints",
    "ValidationResult",
    "FeatureExtractor",
    "ConstraintValidator",
]
