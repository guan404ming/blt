"""Singing Voice Synthesis (SVS) module.

This module provides components for:
- Vocal separation from audio
- Lyrics alignment with audio
- Voice synthesis with new lyrics
- Complete pipeline for cover song generation
"""

from .vocal_separator import VocalSeparator
from .lyrics_aligner import LyricsAligner
from .voice_synthesizer import VoiceSynthesizer
from .pipeline import LyricsTranslationPipeline

__all__ = [
    "VocalSeparator",
    "LyricsAligner",
    "VoiceSynthesizer",
    "LyricsTranslationPipeline",
]
