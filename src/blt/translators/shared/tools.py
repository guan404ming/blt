"""
Tool definitions for translation workflows
"""

from langchain_core.tools import tool


def create_translation_tools(analyzer):
    """
    Create tool functions for lyrics translation

    Args:
        analyzer: LyricsAnalyzer instance

    Returns:
        List of tool functions that can be bound to LLM
    """

    @tool
    def count_syllables(text: str, language: str) -> int:
        """
        Count the number of syllables in the given text.

        Use this to verify your translation has the correct syllable count.

        Args:
            text: The text to count syllables in
            language: The language code (e.g., 'en', 'zh', 'ja')

        Returns:
            The number of syllables in the text
        """
        return analyzer.count_syllables(text, language)

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
    def get_phoneme_ipa(text: str, language: str) -> str:
        """
        Get the IPA (International Phonetic Alphabet) representation of text.

        Use this to understand the phonetic structure of the source text.

        Args:
            text: The text to convert to IPA
            language: The language code (e.g., 'en', 'zh', 'ja')

        Returns:
            The IPA representation
        """
        return analyzer.text_to_ipa(text, language)

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

    return [get_phoneme_ipa, check_phonetic_similarity]
