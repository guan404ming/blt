"""AI-Powered Lyrics Translation module.

This module provides:
- PydanticAI + Gemini-based lyrics translation
- Music constraints extraction and validation
- Chain-of-Thought reasoning for better translations
- Support for multiple languages with syllable/rhyme preservation
"""

from .translator import LyricsTranslator
from .models import LyricTranslation, MusicConstraints, CoTTranslation, ValidationResult
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator

__version__ = "0.2.0"

__all__ = [
    "LyricsTranslator",
    "LyricTranslation",
    "MusicConstraints",
    "CoTTranslation",
    "ValidationResult",
    "FeatureExtractor",
    "ConstraintValidator",
]
