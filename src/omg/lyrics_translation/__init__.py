"""Lyrics translation and cover song generation module.

This module provides two complementary systems:

1. Audio Processing Pipeline (LyricsTranslationPipeline):
   - Vocal separation
   - Lyrics alignment
   - Voice synthesis
   - Audio mixing

2. AI-Powered Lyrics Translation (LyricsTranslator):
   - PydanticAI + Gemini-based translation
   - Music constraints preservation
   - Chain-of-Thought reasoning
"""

# Audio processing components
from .vocal_separator import VocalSeparator
from .lyrics_aligner import LyricsAligner
from .voice_synthesizer import VoiceSynthesizer
from .pipeline import LyricsTranslationPipeline

# AI translation components
from .translator import LyricsTranslator
from .models import LyricTranslation, MusicConstraints, CoTTranslation, ValidationResult
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator

__version__ = "0.2.0"

__all__ = [
    # Audio processing
    "VocalSeparator",
    "LyricsAligner",
    "VoiceSynthesizer",
    "LyricsTranslationPipeline",
    # AI translation
    "LyricsTranslator",
    "LyricTranslation",
    "MusicConstraints",
    "CoTTranslation",
    "ValidationResult",
    "FeatureExtractor",
    "ConstraintValidator",
]
