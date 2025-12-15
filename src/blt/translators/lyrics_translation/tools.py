"""
Tool definitions for constraint-based lyrics translation
Wraps Validator.verify_all_constraints with LLM-friendly tools
"""

from langchain_core.tools import tool
from ..shared.tools import count_syllables


def create_translation_tools(analyzer, validator):
    """
    Create tool functions for lyrics translation

    Args:
        analyzer: LyricsAnalyzer instance
        validator: Validator instance

    Returns:
        List of tool functions that can be bound to LLM
    """

    @tool
    def verify_all_constraints(
        lines: list[str],
        language: str,
        target_syllables: list[int],
        rhyme_scheme: str = "",
        target_patterns: list[list[int]] | None = None,
    ) -> dict:
        """
        Verify all translation constraints at once.

        Use this to check if your translation meets all syllable count, rhyme, and pattern requirements.

        Args:
            lines: List of translated lines
            language: The language code (e.g., 'en-us', 'cmn', 'ja')
            target_syllables: Target syllable count for each line
            rhyme_scheme: Rhyme scheme (e.g., "AABB")
            target_patterns: Optional target syllable patterns

        Returns:
            A dictionary with:
            - syllables: Actual syllable counts
            - syllables_match: Whether syllables match
            - feedback: Detailed feedback on constraint violations
            - passed: Whether all constraints are satisfied
        """
        return validator.verify_all_constraints(
            lines, language, target_syllables, rhyme_scheme, target_patterns
        )

    return [count_syllables, verify_all_constraints]
