"""Singing Voice Synthesis (SVS) module.

This module provides components for:
- Vocal separation from audio
- Lyrics alignment with audio
- Voice synthesis with new lyrics
"""

from .vocal_separator import VocalSeparator
from .lyrics_aligner import LyricsAligner
from .voice_synthesizer import VoiceSynthesizer

__all__ = [
    "VocalSeparator",
    "LyricsAligner",
    "VoiceSynthesizer",
]
