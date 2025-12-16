"""Singing Voice Synthesis (SVS) module.

This module provides components for:
- Vocal separation from audio
- Lyrics alignment with audio
- Voice synthesis with new lyrics
- Lip-sync video generation
"""

from .vocal_separator import VocalSeparator
from .lyrics_aligner import LyricsAligner
from .voice_converter import RetrievalBasedVoiceConverter
from .audio_mixer import AudioMixer
from .video_generator import LipSyncedVideoGenerator, KTVVideoGenerator

__all__ = [
    "VocalSeparator",
    "LyricsAligner",
    "RetrievalBasedVoiceConverter",
    "AudioMixer",
    "LipSyncedVideoGenerator",
    "KTVVideoGenerator",
]
