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
    ) -> dict:
        """
        Verify all constraints at once (most efficient tool).

        Returns: {
            "syllables": [int],
            "syllables_match": bool,
            "rhyme_endings": [str],
            "rhymes_valid": bool,
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

        return {
            "syllables": syllables,
            "syllables_match": syllables_match,
            "rhyme_endings": rhyme_endings,
            "rhymes_valid": rhymes_valid,
            "feedback": feedback,
        }

    def count_syllables(self, text: str, language: str) -> int:
        """Count syllables in text."""
        return self.extractor._count_syllables(text, language)

    def check_rhyme(self, text1: str, text2: str, language: str) -> dict:
        """Check if two texts rhyme. Returns: {"rhymes": bool, "rhyme1": str, "rhyme2": str}"""
        rhyme1 = self.extractor._extract_rhyme_ending(text1, language)
        rhyme2 = self.extractor._extract_rhyme_ending(text2, language)
        rhymes = self._rhymes_with(rhyme1, rhyme2)
        return {"rhymes": rhymes, "rhyme1": rhyme1, "rhyme2": rhyme2}

    # ==================== POST-VALIDATION ====================

    def validate(
        self, translation: LyricTranslation, constraints: MusicConstraints
    ) -> ValidationResult:
        """Simple validation for final result display."""
        # Check syllables
        syllables_match = translation.syllable_counts == constraints.syllable_counts

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

        passed = syllables_match and rhymes_valid
        score = 1.0 if passed else 0.5

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
