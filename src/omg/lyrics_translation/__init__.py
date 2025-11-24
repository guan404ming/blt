"""Lyrics translation and cover song generation module."""

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
