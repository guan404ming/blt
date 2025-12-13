"""
Constraint Validators
Uses LyricsAnalyzer for all core functionality
"""

from .models import (
    LyricTranslation,
    MusicConstraints,
    ValidationResult,
)
from .analyzer import LyricsAnalyzer


class ConstraintValidator:
    """
    Validates lyrics translations against music constraints

    This validator works with LyricsAnalyzer to verify that translations
    meet syllable count, rhyme scheme, and syllable pattern requirements.
    """

    def __init__(self, analyzer: LyricsAnalyzer):
        """
        Initialize validator

        Args:
            analyzer: LyricsAnalyzer instance for core analysis operations
        """
        self.analyzer = analyzer

    # ==================== PUBLIC API (for LLM tools) ====================

    def verify_all_constraints(
        self,
        lines: list[str],
        language: str,
        target_syllables: list[int],
        rhyme_scheme: str = "",
        target_patterns: list[list[int]] | None = None,
    ) -> dict:
        """
        Verify all constraints at once

        Args:
            lines: List of translated lines
            language: Language code
            target_syllables: Target syllable count for each line
            rhyme_scheme: Rhyme scheme (e.g., "AABB")
            target_patterns: Optional target syllable patterns

        Returns:
            dict with verification results and feedback
        """
        # Count syllables
        syllables = [self.analyzer.count_syllables(line, language) for line in lines]
        syllables_match = syllables == target_syllables

        # Extract rhyme endings
        rhyme_endings = [
            self.analyzer.extract_rhyme_ending(line, language) for line in lines
        ]

        # Build feedback (ordered by priority: patterns > syllables > rhymes)
        feedback_parts = []

        # Check syllable patterns if provided (HIGHEST PRIORITY)
        patterns_match = True
        syllable_patterns = None
        if target_patterns:
            syllable_patterns = self.analyzer.get_syllable_patterns(lines, language)
            patterns_match = syllable_patterns == target_patterns

            if not patterns_match:
                pattern_feedback = self._build_pattern_feedback(
                    syllable_patterns, target_patterns
                )
                if pattern_feedback:
                    feedback_parts.append(
                        "🚨 CRITICAL: SYLLABLE PATTERN MISMATCHES (FIX FIRST!):\n\n"
                        + "\n\n".join(pattern_feedback)
                    )

        # Syllable feedback (SECOND PRIORITY)
        if not syllables_match:
            mismatches = self._build_syllable_feedback(syllables, target_syllables)
            if mismatches:
                feedback_parts.append(
                    "⚠️  SYLLABLE COUNT MISMATCHES:\n" + "\n".join(mismatches)
                )

        # Check rhyme scheme (LOWEST PRIORITY)
        rhymes_valid = True
        if rhyme_scheme:
            rhymes_valid, rhyme_issues = self._check_rhyme_scheme(
                rhyme_endings, rhyme_scheme, language
            )
            if rhyme_issues:
                feedback_parts.append(
                    "ℹ️  Rhyme issues (optional):\n" + "\n".join(rhyme_issues)
                )

        # Combine feedback
        feedback = (
            "\n\n".join(feedback_parts)
            if feedback_parts
            else "All constraints satisfied!"
        )

        result = {
            "syllables": syllables,
            "syllables_match": syllables_match,
            "rhyme_endings": rhyme_endings,
            "rhymes_valid": rhymes_valid,
            "feedback": feedback,
        }

        if target_patterns:
            result["syllable_patterns"] = syllable_patterns
            result["patterns_match"] = patterns_match

        return result

    def validate(
        self,
        translation: LyricTranslation,
        constraints: MusicConstraints,
        language: str,
    ) -> ValidationResult:
        """
        Validate translation result

        Args:
            translation: Translation to validate
            constraints: Target constraints
            language: Language code for rhyme checking

        Returns:
            ValidationResult with pass/fail and score
        """
        # Check syllables
        syllables_match = translation.syllable_counts == constraints.syllable_counts

        # Check syllable patterns
        patterns_match = True
        if constraints.syllable_patterns and translation.syllable_patterns:
            patterns_match = (
                translation.syllable_patterns == constraints.syllable_patterns
            )

        # Check rhymes
        rhymes_valid = True
        if constraints.rhyme_scheme:
            rhymes_valid, _ = self._check_rhyme_scheme(
                translation.rhyme_endings, constraints.rhyme_scheme, language
            )

        passed = syllables_match and rhymes_valid and patterns_match

        # Calculate score (patterns most important, rhymes least important)
        score_components = []
        if patterns_match:
            score_components.append(0.6)
        if syllables_match:
            score_components.append(0.3)
        if rhymes_valid or not constraints.rhyme_scheme:
            score_components.append(0.1)

        score = sum(score_components)

        return ValidationResult(passed=passed, errors=[], score=score)

    # ==================== PRIVATE HELPERS ====================

    def _build_syllable_feedback(
        self, actual: list[int], target: list[int]
    ) -> list[str]:
        """Build feedback for syllable mismatches"""
        mismatches = []
        for i, (act, tgt) in enumerate(zip(actual, target)):
            if act != tgt:
                diff = act - tgt
                if diff > 0:
                    mismatches.append(
                        f"Line {i + 1}: {act} syllables (need {diff} fewer)"
                    )
                else:
                    mismatches.append(
                        f"Line {i + 1}: {act} syllables (need {abs(diff)} more)"
                    )
        return mismatches

    def _build_pattern_feedback(
        self, actual: list[list[int]], target: list[list[int]]
    ) -> list[str]:
        """Build feedback for pattern mismatches"""
        pattern_mismatches = []
        for i, (act, tgt) in enumerate(zip(actual, target)):
            if act != tgt:
                actual_str = "[" + ", ".join(str(s) for s in act) + "]"
                target_str = "[" + ", ".join(str(s) for s in tgt) + "]"
                actual_total = sum(act)
                target_total = sum(tgt)

                details = [
                    f"Line {i + 1}:",
                    f"  Actual:  {actual_str} (total: {actual_total} syllables)",
                    f"  Target:  {target_str} (total: {target_total} syllables)",
                ]
                pattern_mismatches.append("\n".join(details))
        return pattern_mismatches

    def _check_rhyme_scheme(
        self, rhyme_endings: list[str], rhyme_scheme: str, language: str
    ) -> tuple[bool, list[str]]:
        """Check if rhyme endings match rhyme scheme"""
        rhymes_valid = True
        rhyme_issues = []

        rhyme_groups = {}
        for i, label in enumerate(rhyme_scheme):
            rhyme_groups.setdefault(label, []).append(i)

        for label, indices in rhyme_groups.items():
            if len(indices) > 1:
                base = rhyme_endings[indices[0]]
                for idx in indices[1:]:
                    if not self.analyzer.check_rhyme(
                        base, rhyme_endings[idx], language
                    ):
                        rhymes_valid = False
                        rhyme_issues.append(
                            f"Lines {indices[0] + 1} and {idx + 1} (group '{label}'): "
                            f"'{rhyme_endings[indices[0]]}' vs '{rhyme_endings[idx]}' don't rhyme"
                        )

        return rhymes_valid, rhyme_issues


# ==================== SORAMIMI VALIDATOR ====================


class SoramimiValidator:
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

    def compare_ipa(
        self,
        source_lines: list[str],
        target_lines: list[str],
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Compare IPA of source and target lyrics

        Args:
            source_lines: Source lyrics lines
            target_lines: Target lyrics lines
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            dict with IPA comparisons and similarity scores
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

    def validate_single_line(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Validate a single line translation

        Args:
            source_text: Source text
            target_text: Target text
            source_lang: Source language
            target_lang: Target language

        Returns:
            dict with IPA and similarity
        """
        src_ipa = self.analyzer.text_to_ipa(source_text, source_lang)
        tgt_ipa = self.analyzer.text_to_ipa(target_text, target_lang)

        # For Chinese, use direct text comparison via pinyin instead of IPA
        is_chinese = target_lang in ("cmn", "zh", "zh-cn", "zh-tw")
        if is_chinese:
            similarity = self.analyzer.calculate_ipa_similarity(
                source_text, target_text, is_chinese=True
            )
        else:
            similarity = self.analyzer.calculate_ipa_similarity(src_ipa, tgt_ipa)

        return {
            "source_ipa": src_ipa,
            "target_ipa": tgt_ipa,
            "similarity": similarity,
            "passed": similarity >= self.similarity_threshold,
        }
