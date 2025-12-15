"""
Tool definitions for soramimi (phonetic) translation
"""

from langchain_core.tools import tool
from ..shared.tools import text_to_ipa


def create_soramimi_tools(analyzer, validator):
    """
    Create tool functions for soramimi translation

    Args:
        analyzer: LyricsAnalyzer instance
        validator: SoramimiValidator instance

    Returns:
        List of tool functions that can be bound to LLM
    """

    @tool
    def check_phonetic_similarity(
        source_text: str, target_text: str, source_lang: str, target_lang: str
    ) -> dict:
        """
        Check phonetic similarity between source and target text.

        Use this to verify that your soramimi mapping produces phonetically similar results.

        Args:
            source_text: The original source text
            target_text: The proposed soramimi translation
            source_lang: The source language code
            target_lang: The target language code

        Returns:
            A dictionary with similarity score and details
        """
        validation = validator.compare_ipa(
            [source_text], [target_text], source_lang, target_lang
        )
        return {
            "similarity_score": validation["overall_similarity"],
            "source_ipa": validation["source_ipas"][0],
            "target_ipa": validation["target_ipas"][0],
            "details": f"Similarity: {validation['overall_similarity']:.1%}",
        }

    return [text_to_ipa, check_phonetic_similarity]
