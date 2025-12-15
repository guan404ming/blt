"""Validator for constraint-based lyrics translations"""

from langchain_core.tools import tool
from ..shared import LyricsAnalyzer


class Validator:
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
        pattern_similarity_score = 1.0
        syllable_patterns = None
        if target_patterns:
            syllable_patterns = self.analyzer.get_syllable_patterns(lines, language)
            patterns_match = syllable_patterns == target_patterns

            if not patterns_match:
                # Calculate pattern alignment and similarity score
                alignments = []
                total_similarity = 0.0
                for i, (actual, target) in enumerate(
                    zip(syllable_patterns, target_patterns)
                ):
                    alignment = self.analyzer.analyze_pattern_alignment(target, actual)
                    alignments.append((i, alignment))
                    total_similarity += alignment["similarity"]

                pattern_similarity_score = (
                    total_similarity / len(alignments) if alignments else 0.0
                )

                pattern_feedback = self._build_pattern_feedback_fuzzy(alignments)
                if pattern_feedback:
                    # Adjust severity based on fuzzy similarity
                    if pattern_similarity_score >= 0.8:
                        severity = "✓ PATTERN ACCEPTABLE (fuzzy match):"
                    elif pattern_similarity_score >= 0.6:
                        severity = "⚠️  PATTERN CLOSE (minor adjustments needed):"
                    else:
                        severity = "🚨 CRITICAL: PATTERN MISMATCH (significant adjustments needed):"

                    feedback_parts.append(
                        f"{severity}\n\n" + "\n\n".join(pattern_feedback)
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
            result["pattern_similarity_score"] = pattern_similarity_score

        return result

    # ==================== TOOLS FOR LLM ====================

    def get_tools(self):
        """Get LLM-callable tools for this validator"""
        validator = self

        @tool
        def verify_all_constraints_tool(
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
                A dictionary with constraint verification results and feedback
            """
            return validator.verify_all_constraints(
                lines, language, target_syllables, rhyme_scheme, target_patterns
            )

        return [verify_all_constraints_tool]

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

    def _build_pattern_feedback_fuzzy(
        self, alignments: list[tuple[int, dict]]
    ) -> list[str]:
        """Build feedback for pattern mismatches using fuzzy alignment analysis"""
        pattern_feedback = []
        for line_idx, alignment in alignments:
            if not alignment["matches"]:
                similarity = alignment["similarity"]
                differences = alignment.get("differences", [])

                # Reconstruct patterns from differences analysis
                if differences:
                    # Pattern info is in the differences
                    details = [f"Line {line_idx + 1}: {similarity:.0%} similar"]

                    # Show target vs current
                    target_vals = [d["target_syllables"] for d in differences]
                    current_vals = [d["current_syllables"] for d in differences]

                    target_str = "[" + ", ".join(str(v) for v in target_vals) + "]"
                    current_str = "[" + ", ".join(str(v) for v in current_vals) + "]"

                    details.append(f"  Actual:  {current_str}")
                    details.append(f"  Target:  {target_str}")

                    # Add specific suggestions if available
                    suggestions = alignment.get("suggestions", [])
                    if suggestions:
                        details.append("  Suggestions:")
                        for suggestion in suggestions[:2]:  # Top 2 suggestions
                            details.append(f"    • {suggestion}")

                    pattern_feedback.append("\n".join(details))

        return pattern_feedback

    def _check_rhyme_scheme(
        self, rhyme_endings: list[str], rhyme_scheme: str, language: str
    ) -> tuple[bool, list[str]]:
        """Check if rhyme endings match rhyme scheme"""
        rhymes_valid = True
        rhyme_issues = []

        # Handle empty rhyme_endings
        if not rhyme_endings or not rhyme_scheme:
            return True, []

        # Handle mismatch between rhyme_endings and rhyme_scheme length
        if len(rhyme_endings) != len(rhyme_scheme):
            return False, [
                f"Rhyme scheme length mismatch: expected {len(rhyme_scheme)}, got {len(rhyme_endings)}"
            ]

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
