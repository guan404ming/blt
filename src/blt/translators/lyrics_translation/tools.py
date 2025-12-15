"""
Tool definitions for constraint-based lyrics translation
"""

from langchain_core.tools import tool
from ..shared.tools import count_syllables


def create_translation_tools(analyzer):
    """
    Create tool functions for lyrics translation

    Args:
        analyzer: LyricsAnalyzer instance

    Returns:
        List of tool functions that can be bound to LLM
    """

    @tool
    def verify_translation(
        translation: str, target_syllables: int, language: str
    ) -> dict:
        """
        Verify if a translation has the target number of syllables.

        Use this to check if your translation meets the syllable requirement.

        Args:
            translation: The translated text to verify
            target_syllables: The target syllable count
            language: The language code (e.g., 'en', 'zh', 'ja')

        Returns:
            A dictionary with:
            - passed: True if syllable count matches target
            - actual: The actual syllable count
            - feedback: Guidance if the count doesn't match
        """
        actual = analyzer.count_syllables(translation, language)
        passed = actual == target_syllables
        diff = actual - target_syllables

        if diff > 0:
            feedback = f"Too many syllables! Need {diff} fewer."
        elif diff < 0:
            feedback = f"Too few syllables! Need {abs(diff)} more."
        else:
            feedback = "Perfect! Syllable count matches."

        return {
            "passed": passed,
            "actual": actual,
            "target": target_syllables,
            "feedback": feedback,
        }

    return [count_syllables, verify_translation]
