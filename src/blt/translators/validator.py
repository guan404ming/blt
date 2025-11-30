"""
Constraint Validator - LLM Toolbox for lyrics translation validation
"""

from .models import LyricTranslation, MusicConstraints, ValidationResult
from .feature_extractor import FeatureExtractor


class ConstraintValidator:
    """Provides validation tools for LLM during translation"""

    def __init__(self, target_lang: str = "Chinese"):
        self.target_lang = target_lang
        self.extractor = FeatureExtractor(target_lang=target_lang)

    # ==================== LLM TOOLS ====================

    def verify_all_constraints(
        self,
        lines: list[str],
        language: str,
        target_syllables: list[int],
        rhyme_scheme: str = "",
        target_patterns: list[list[int]] | None = None,
    ) -> dict:
        """
        Verify all constraints at once (most efficient tool).

        Args:
            lines: List of translated lines
            language: Language code
            target_syllables: Target syllable count for each line
            rhyme_scheme: Rhyme scheme (e.g., "AABB")
            target_patterns: Optional target syllable patterns, e.g., [[1, 1, 3], [1, 2, 1]]

        Returns: {
            "syllables": [int],
            "syllables_match": bool,
            "rhyme_endings": [str],
            "rhymes_valid": bool,
            "syllable_patterns": [[int]] (if target_patterns provided),
            "patterns_match": bool (if target_patterns provided),
            "feedback": str  # Improvement suggestions
        }
        """
        # Count syllables
        syllables = [self.extractor._count_syllables(line, language) for line in lines]
        syllables_match = syllables == target_syllables

        # Extract rhyme endings
        rhyme_endings = [
            self.extractor._extract_rhyme_ending(line, language) for line in lines
        ]

        # Build feedback
        feedback_parts = []

        # Syllable feedback
        if not syllables_match:
            mismatches = []
            for i, (actual, target) in enumerate(zip(syllables, target_syllables)):
                if actual != target:
                    diff = actual - target
                    if diff > 0:
                        mismatches.append(
                            f"Line {i + 1}: {actual} syllables (need {diff} fewer)"
                        )
                    else:
                        mismatches.append(
                            f"Line {i + 1}: {actual} syllables (need {abs(diff)} more)"
                        )
            if mismatches:
                feedback_parts.append("SYLLABLE MISMATCHES:\n" + "\n".join(mismatches))

        # Check syllable patterns if provided (uses batch LLM call)
        patterns_match = True
        syllable_patterns = None
        if target_patterns:
            syllable_patterns = self.extractor._get_syllable_patterns(lines, language)
            patterns_match = syllable_patterns == target_patterns

            if not patterns_match:
                pattern_mismatches = []
                for i, (actual, target) in enumerate(
                    zip(syllable_patterns, target_patterns)
                ):
                    if actual != target:
                        actual_str = "[" + ", ".join(str(s) for s in actual) + "]"
                        target_str = "[" + ", ".join(str(s) for s in target) + "]"
                        actual_total = sum(actual)
                        target_total = sum(target)

                        mismatch_details = []
                        mismatch_details.append(f"Line {i + 1}:")
                        mismatch_details.append(
                            f"  Actual:  {actual_str} (total: {actual_total} syllables)"
                        )
                        mismatch_details.append(
                            f"  Target:  {target_str} (total: {target_total} syllables)"
                        )

                        pattern_mismatches.append("\n".join(mismatch_details))

                if pattern_mismatches:
                    feedback_parts.append(
                        "SYLLABLE PATTERN MISMATCHES:\n\n"
                        + "\n\n".join(pattern_mismatches)
                    )

        # Check rhyme scheme
        rhymes_valid = True
        rhyme_issues = []
        if rhyme_scheme:
            rhyme_groups = {}
            for i, label in enumerate(rhyme_scheme):
                rhyme_groups.setdefault(label, []).append(i)

            for label, indices in rhyme_groups.items():
                if len(indices) > 1:
                    base = rhyme_endings[indices[0]]
                    for idx in indices[1:]:
                        if not self._rhymes_with(base, rhyme_endings[idx]):
                            rhymes_valid = False
                            rhyme_issues.append(
                                f"Lines {indices[0] + 1} and {idx + 1} (group '{label}'): '{rhyme_endings[indices[0]]}' vs '{rhyme_endings[idx]}' don't rhyme"
                            )

            if rhyme_issues:
                feedback_parts.append("RHYME ISSUES:\n" + "\n".join(rhyme_issues))

        # Combine feedback
        if feedback_parts:
            feedback = "\n\n".join(feedback_parts)
        else:
            feedback = "All constraints satisfied!"

        result = {
            "syllables": syllables,
            "syllables_match": syllables_match,
            "rhyme_endings": rhyme_endings,
            "rhymes_valid": rhymes_valid,
            "feedback": feedback,
        }

        # Add syllable pattern results if checked
        if target_patterns:
            result["syllable_patterns"] = syllable_patterns
            result["patterns_match"] = patterns_match

        return result

    def count_syllables(self, text: str, language: str) -> int:
        """Count syllables in text."""
        return self.extractor._count_syllables(text, language)

    def check_rhyme(self, text1: str, text2: str, language: str) -> dict:
        """Check if two texts rhyme. Returns: {"rhymes": bool, "rhyme1": str, "rhyme2": str}"""
        rhyme1 = self.extractor._extract_rhyme_ending(text1, language)
        rhyme2 = self.extractor._extract_rhyme_ending(text2, language)
        rhymes = self._rhymes_with(rhyme1, rhyme2)
        return {"rhymes": rhymes, "rhyme1": rhyme1, "rhyme2": rhyme2}

    def verify_syllable_pattern(
        self, lines: list[str], language: str, target_patterns: list[list[int]]
    ) -> dict:
        """
        Verify that syllable patterns match exactly (uses batch LLM call).

        Args:
            lines: List of translated lines
            language: Language code
            target_patterns: Target syllable pattern for each line, e.g., [[1, 1, 3], [1, 2, 1]]

        Returns: {
            "syllable_patterns": [[int]],
            "patterns_match": bool,
            "feedback": str
        }
        """
        # Batch process all lines in one LLM call
        syllable_patterns = self.extractor._get_syllable_patterns(lines, language)
        patterns_match = syllable_patterns == target_patterns

        feedback_parts = []
        if not patterns_match:
            mismatches = []
            for i, (actual, target) in enumerate(
                zip(syllable_patterns, target_patterns)
            ):
                if actual != target:
                    actual_str = "[" + ", ".join(str(s) for s in actual) + "]"
                    target_str = "[" + ", ".join(str(s) for s in target) + "]"
                    actual_total = sum(actual)
                    target_total = sum(target)

                    mismatch_details = []
                    mismatch_details.append(f"Line {i + 1}:")
                    mismatch_details.append(
                        f"  Actual:  {actual_str} (total: {actual_total} syllables, {len(actual)} words)"
                    )
                    mismatch_details.append(
                        f"  Target:  {target_str} (total: {target_total} syllables, {len(target)} words)"
                    )
                    mismatches.append("\n".join(mismatch_details))

        feedback = (
            "\n\n".join(feedback_parts)
            if feedback_parts
            else "Syllable patterns match perfectly!"
        )

        return {
            "syllable_patterns": syllable_patterns,
            "patterns_match": patterns_match,
            "feedback": feedback,
        }

    # ==================== POST-VALIDATION ====================

    def validate(
        self, translation: LyricTranslation, constraints: MusicConstraints
    ) -> ValidationResult:
        """Simple validation for final result display."""
        # Check syllables
        syllables_match = translation.syllable_counts == constraints.syllable_counts

        # Check syllable patterns if both exist
        patterns_match = True
        if (
            constraints.syllable_patterns
            and translation.syllable_patterns
            and len(constraints.syllable_patterns) == len(translation.syllable_patterns)
        ):
            patterns_match = (
                translation.syllable_patterns == constraints.syllable_patterns
            )

        # Check rhymes if needed
        rhymes_valid = True
        if constraints.rhyme_scheme:
            rhyme_groups = {}
            for i, label in enumerate(constraints.rhyme_scheme):
                rhyme_groups.setdefault(label, []).append(i)

            for indices in rhyme_groups.values():
                if len(indices) > 1:
                    base = translation.rhyme_endings[indices[0]]
                    for idx in indices[1:]:
                        if not self._rhymes_with(base, translation.rhyme_endings[idx]):
                            rhymes_valid = False
                            break

        passed = syllables_match and rhymes_valid and patterns_match

        # Calculate score based on what matched
        score_components = []
        if syllables_match:
            score_components.append(0.5)  # 50% for syllable count
        if rhymes_valid or not constraints.rhyme_scheme:
            score_components.append(0.3)  # 30% for rhymes
        if patterns_match:
            score_components.append(0.2)  # 20% for syllable patterns

        score = sum(score_components)

        return ValidationResult(
            passed=passed,
            errors=[],
            score=score,
        )

    # ==================== HELPERS ====================

    def _rhymes_with(self, rhyme1: str, rhyme2: str) -> bool:
        """Check if two rhyme endings match."""
        if not rhyme1 or not rhyme2:
            return False
        return rhyme1 == rhyme2 or rhyme1 in rhyme2 or rhyme2 in rhyme1
