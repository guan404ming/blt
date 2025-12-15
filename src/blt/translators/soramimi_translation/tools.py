"""
Tool definitions for soramimi (phonetic) translation
Wraps Validator.verify_all_constraints with LLM-friendly tools
"""

from langchain_core.tools import tool
from ..shared.tools import text_to_ipa


def create_soramimi_tools(analyzer, validator):
    """
    Create tool functions for soramimi (phonetic) translation

    Args:
        analyzer: LyricsAnalyzer instance for IPA conversion
        validator: Validator instance for phonetic similarity checking

    Returns:
        List of tool functions that can be bound to LLM
    """

    @tool
    def verify_all_constraints(
        source_lines: list[str],
        target_lines: list[str],
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Verify all phonetic constraints by comparing IPA similarity.

        Use this to check if your soramimi mapping produces phonetically similar results.

        Args:
            source_lines: Source lyrics lines
            target_lines: Target lyrics lines (translations to verify)
            source_lang: Source language code (e.g., 'en-us', 'cmn', 'ja')
            target_lang: Target language code (e.g., 'en-us', 'cmn', 'ja')

        Returns:
            A dictionary with:
            - source_ipas: IPA transcriptions of source lines
            - target_ipas: IPA transcriptions of target lines
            - similarities: Similarity score for each line (0-1)
            - overall_similarity: Overall similarity across all lines
            - passed: Whether all lines meet similarity threshold
            - feedback: Detailed feedback on constraint violations
        """
        return validator.verify_all_constraints(
            source_lines, target_lines, source_lang, target_lang
        )

    return [text_to_ipa, verify_all_constraints]
