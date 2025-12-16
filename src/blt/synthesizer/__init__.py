"""Singing Voice Synthesis (SVS) module.

This module provides components for:
- Vocal separation from audio
- Lyrics alignment with audio
- Voice synthesis with new lyrics
- Lip-sync video generation
"""

from .vocal_separator import VocalSeparator
from .lyrics_aligner import LyricsAligner, WhisperLyricsAligner
from .voice_synthesizer import VoiceSynthesizer
from .voice_converter import RetrievalBasedVoiceConverter
from .wav2lip import Wav2Lip

__all__ = [
    "VocalSeparator",
    "LyricsAligner",
    "WhisperLyricsAligner",
    "VoiceSynthesizer",
    "RetrievalBasedVoiceConverter",
    "Wav2Lip",
]
