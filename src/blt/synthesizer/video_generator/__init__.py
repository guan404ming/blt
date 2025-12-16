"""Video generation module.

This module provides two video generators:
- LipSyncedVideoGenerator: Generates lip-synced videos using Wav2Lip
- KTVVideoGenerator: Generates KTV (karaoke) videos with dual-track subtitles
"""

from .lipsync_video_generator import LipSyncedVideoGenerator
from .ktv_video_generator import KTVVideoGenerator

__all__ = [
    "LipSyncedVideoGenerator",
    "KTVVideoGenerator",
]
