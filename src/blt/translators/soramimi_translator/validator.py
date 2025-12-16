"""Validator for soramimi (phonetic) translations"""

from langchain_core.tools import tool
from ..shared import LyricsAnalyzer


class Validator:
    """
    Validates soramimi (phonetic) translations by comparing IPA similarity

    This validator works with LyricsAnalyzer to verify that soramimi translations
    maintain phonetic similarity to the source text by comparing IPA transcriptions
    and using feature-based distance calculations.
    """

    def __init__(self, analyzer: LyricsAnalyzer, similarity_threshold: float = 0.6):
        """
        Initialize validator

        Args:
            analyzer: LyricsAnalyzer instance for IPA conversion and similarity calculations
            similarity_threshold: Minimum similarity score to pass validation (0.0-1.0)
        """
        self.analyzer = analyzer
        self.similarity_threshold = similarity_threshold

    # ==================== PUBLIC API (for LLM tools) ====================

    def verify_all_constraints(
        self,
        source_lines: list[str],
        target_lines: list[str],
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Verify all phonetic constraints by comparing IPA similarity

        Args:
            source_lines: Source lyrics lines
            target_lines: Target lyrics lines (translations to verify)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            dict with:
            - source_ipas: IPA transcriptions of source lines
            - target_ipas: IPA transcriptions of target lines
            - similarities: Similarity score for each line (0-1)
            - overall_similarity: Overall similarity across all lines
            - passed: Whether all lines meet similarity threshold
            - feedback: Detailed feedback on constraint violations
        """
        source_ipas = []
        target_ipas = []
        similarities = []
        feedback_parts = []

        # Check if target is Chinese
        is_chinese = target_lang in ("cmn", "zh", "zh-cn", "zh-tw")

        for i, (src, tgt) in enumerate(zip(source_lines, target_lines)):
            src_ipa = self.analyzer.text_to_ipa(src, source_lang)
            tgt_ipa = self.analyzer.text_to_ipa(tgt, target_lang)

            source_ipas.append(src_ipa)
            target_ipas.append(tgt_ipa)

            # For Chinese, use direct text comparison via pinyin
            if is_chinese:
                similarity = self.analyzer.calculate_ipa_similarity(
                    src, tgt, is_chinese=True
                )
            else:
                similarity = self.analyzer.calculate_ipa_similarity(src_ipa, tgt_ipa)
            similarities.append(similarity)

            if similarity < self.similarity_threshold:
                feedback_parts.append(
                    f"Line {i + 1}: {similarity:.1%} similarity (need >= {self.similarity_threshold:.0%})\n"
                    f"  Source IPA: {src_ipa}\n"
                    f"  Target IPA: {tgt_ipa}"
                )

        overall_similarity = (
            sum(similarities) / len(similarities) if similarities else 0
        )
        passed = all(s >= self.similarity_threshold for s in similarities)

        feedback = (
            "\n\n".join(feedback_parts)
            if feedback_parts
            else f"All lines meet similarity threshold ({self.similarity_threshold:.0%})!"
        )

        return {
            "source_ipas": source_ipas,
            "target_ipas": target_ipas,
            "similarities": similarities,
            "overall_similarity": overall_similarity,
            "passed": passed,
            "feedback": feedback,
        }

    # ==================== TOOLS FOR LLM ====================

    def get_tools(self):
        """Get LLM-callable tools for this validator"""
        validator = self

        @tool
        def verify_all_constraints_tool(
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
                A dictionary with constraint verification results and feedback
            """
            return validator.verify_all_constraints(
                source_lines, target_lines, source_lang, target_lang
            )

        return [verify_all_constraints_tool]
