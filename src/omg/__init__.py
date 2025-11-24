"""Optimized Music Generation"""

__version__ = "0.2.0"

from omg.base import generate_music

# Import Synthesizer module (Singing Voice Synthesis)
from omg import synthesizer

# Import Translators module
from omg import translators

# Import Pipeline module
from omg import pipeline

__all__ = [
    "generate_music",
    "synthesizer",
    "translators",
    "pipeline",
]
