"""
LangGraph state and graph definitions for translation workflows

This module organizes the translation agent components following the ReAct pattern:
- states.py: State definitions and initialization functions
- tools.py: Tool definitions for LLM tool use
- prompts.py: System and user prompt templates
- lyrics_translation.py: Lyrics translation graph builder
- soramimi_translation.py: Soramimi translation graph builder
"""

from .states import (
    LyricsTranslationState,
    SoramimiMappingState,
    create_lyrics_translation_initial_state,
    create_soramimi_mapping_initial_state,
)
from .lyrics_translation import build_lyrics_translation_graph
from .soramimi_translation import build_soramimi_mapping_graph
from .tools import create_translation_tools, create_soramimi_tools
from .prompts import (
    get_lyrics_translation_system_prompt,
    get_lyrics_translation_user_prompt,
    get_soramimi_system_prompt,
    get_soramimi_user_prompt,
)

__all__ = [
    # State management
    "LyricsTranslationState",
    "SoramimiMappingState",
    "create_lyrics_translation_initial_state",
    "create_soramimi_mapping_initial_state",
    # Graph builders
    "build_lyrics_translation_graph",
    "build_soramimi_mapping_graph",
    # Tools
    "create_translation_tools",
    "create_soramimi_tools",
    # Prompts
    "get_lyrics_translation_system_prompt",
    "get_lyrics_translation_user_prompt",
    "get_soramimi_system_prompt",
    "get_soramimi_user_prompt",
]
