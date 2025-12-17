"""Validator for Gemini lyrics translations"""

import logging
from ..shared import (
    LyricsAnalyzer,
    count_syllables,
    extract_rhyme_ending,
    detect_rhyme_scheme,
    get_syllable_patterns,
)

logger = logging.getLogger(__name__)


class Validator:
    """Validates Gemini translations against music constraints

    Uses shared constraint tools to verify:
    - Syllable counts match target
    - Rhyme scheme matches target
    - Syllable patterns match target
    """

    def __init__(self, analyzer: LyricsAnalyzer = None):
        """Initialize validator

        Args:
            analyzer: LyricsAnalyzer instance (creates new if None)
        """
        self.analyzer = analyzer or LyricsAnalyzer()

    def verify_all_constraints(
        self,
        lines: list[str],
        language: str,
        target_syllables: list[int],
        rhyme_scheme: str = "",
        target_patterns: list[list[int]] = None,
    ) -> dict:
        """Verify all constraints at once

        Args:
            lines: List of translated lines
            language: Language code (e.g., 'zh-tw')
            target_syllables: Target syllable count for each line
            rhyme_scheme: Target rhyme scheme (e.g., 'ABCDAECDD')
            target_patterns: Target syllable patterns per line

        Returns:
            Dict with verification results:
            - 'passed': bool - All constraints passed
            - 'syllables': dict - Syllable verification details
            - 'rhyme': dict - Rhyme scheme verification details
            - 'patterns': dict - Syllable pattern verification details
            - 'overall_score': float - Overall match score (0-1)
        """
        results = {}
        scores = []

        # Check syllables
        if target_syllables:
            actual_syllables = []
            for line in lines:
                count = count_syllables.invoke(
                    {
                        "text": line,
                        "language": language,
                    }
                )
                actual_syllables.append(count)

            syllables_match = actual_syllables == target_syllables
            results["syllables"] = {
                "passed": syllables_match,
                "target": target_syllables,
                "actual": actual_syllables,
            }
            scores.append(1.0 if syllables_match else 0.0)
            logger.info(f"Syllables check: {syllables_match}")

        # Check rhyme scheme
        if rhyme_scheme:
            rhyme_endings = []
            for line in lines:
                ending = extract_rhyme_ending.invoke(
                    {
                        "text": line,
                        "language": language,
                    }
                )
                rhyme_endings.append(ending)

            actual_scheme = detect_rhyme_scheme.invoke(
                {
                    "lines": lines,
                    "language": language,
                }
            )

            scheme_match = actual_scheme == rhyme_scheme
            results["rhyme"] = {
                "passed": scheme_match,
                "target": rhyme_scheme,
                "actual": actual_scheme,
                "endings": rhyme_endings,
            }
            scores.append(1.0 if scheme_match else 0.0)
            logger.info(
                f"Rhyme scheme check: {scheme_match} (target={rhyme_scheme}, actual={actual_scheme})"
            )

        # Check syllable patterns
        if target_patterns:
            actual_patterns = get_syllable_patterns.invoke(
                {
                    "lines": lines,
                    "language": language,
                }
            )

            patterns_match = actual_patterns == target_patterns
            results["patterns"] = {
                "passed": patterns_match,
                "target": target_patterns,
                "actual": actual_patterns,
            }
            scores.append(1.0 if patterns_match else 0.0)
            logger.info(f"Syllable patterns check: {patterns_match}")

        # Calculate overall results
        overall_score = sum(scores) / len(scores) if scores else 1.0
        all_passed = all(r.get("passed", True) for r in results.values())

        return {
            "passed": all_passed,
            "overall_score": overall_score,
            "syllables": results.get("syllables"),
            "rhyme": results.get("rhyme"),
            "patterns": results.get("patterns"),
        }
