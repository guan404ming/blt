"""
Lyrics Translation System
基於 PydanticAI + Gemini 2.0 Flash 的可唱歌詞翻譯系統
"""

from .models import LyricTranslation, MusicConstraints, CoTTranslation
from .translator import LyricsTranslator
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator

__version__ = "0.1.0"

__all__ = [
    "LyricTranslation",
    "MusicConstraints",
    "CoTTranslation",
    "LyricsTranslator",
    "FeatureExtractor",
    "ConstraintValidator",
]
