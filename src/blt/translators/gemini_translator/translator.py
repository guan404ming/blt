"""Gemini Lyrics Translator - Wrapper for backward compatibility

The main implementation is in agent.py (GeminiTranslationAgent).
This module provides GeminiTranslator as an alias for backward compatibility.
"""

from .agent import GeminiTranslationAgent

# Alias for backward compatibility
GeminiTranslator = GeminiTranslationAgent

__all__ = ["GeminiTranslator", "GeminiTranslationAgent"]
