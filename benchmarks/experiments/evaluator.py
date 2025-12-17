"""
Translation Quality Evaluator

Evaluates translations on three key metrics:
1. Syllable Count Accuracy - How well syllable counts are preserved
2. Syllable Pattern Similarity - How well rhythm patterns match
3. Rhyme Preservation - How well rhyme schemes are maintained
"""

from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from blt.translators import LyricsAnalyzer


class EvaluationMetrics(TypedDict):
    """Evaluation metrics for a translation

    Three core metrics:
    - SER: Syllable Error Rate (edit distance normalized)
    - SCRE: Syllable Count Relative Error (average relative error per line)
    - ARI: Adjusted Rand Index (rhyme clustering agreement)
    """

    # Syllable Error Rate (SER) - edit distance between syllable sequences
    ser: float  # EditDistance(actual, target) / n

    # Syllable Count Relative Error (SCRE) - average relative error in syllable counts
    scre: float  # Average |target - actual| / target per line

    # Adjusted Rand Index (ARI) - rhyme clustering agreement
    ari: float  # [-1, 1] measures agreement of rhyme clustering (1=perfect, 0=random, -1=opposite)


class TranslationEvaluator:
    """
    Evaluates translation quality against source constraints

    This evaluator measures how well a translation preserves:
    - Syllable counts from source
    - Syllable patterns (rhythm)
    - Rhyme schemes
    """

    def __init__(self, analyzer: "LyricsAnalyzer | None" = None):
        """
        Initialize evaluator

        Args:
            analyzer: LyricsAnalyzer instance (creates new if None)
        """
        if analyzer is None:
            from blt.translators import LyricsAnalyzer as Analyzer
            self.analyzer = Analyzer()
        else:
            self.analyzer = analyzer

    def evaluate(
        self,
        source_lines: list[str],
        source_lang: str,
        translated_lines: list[str],
        target_lang: str,
    ) -> EvaluationMetrics:
        """
        Evaluate translation quality

        Args:
            source_lines: Original lyrics lines
            source_lang: Source language code (e.g., 'cmn', 'en-us', 'ja')
            translated_lines: Translated lyrics lines
            target_lang: Target language code

        Returns:
            EvaluationMetrics with all scores
        """
        # Extract source constraints
        source_syllables = [
            self.analyzer.count_syllables(line, source_lang)
            for line in source_lines
        ]
        source_rhyme_scheme = self.analyzer.detect_rhyme_scheme(source_lines, source_lang)

        # Analyze translation
        target_syllables = [
            self.analyzer.count_syllables(line, target_lang)
            for line in translated_lines
        ]
        target_rhyme_scheme = self.analyzer.detect_rhyme_scheme(translated_lines, target_lang)

        # Calculate metrics
        return self._calculate_metrics(
            source_syllables=source_syllables,
            target_syllables=target_syllables,
            source_rhyme_scheme=source_rhyme_scheme,
            target_rhyme_scheme=target_rhyme_scheme,
        )

    def _calculate_metrics(
        self,
        source_syllables: list[int],
        target_syllables: list[int],
        source_rhyme_scheme: str,
        target_rhyme_scheme: str,
    ) -> EvaluationMetrics:
        """Calculate three core evaluation metrics"""

        # 1. SER (Syllable Error Rate) - Edit distance between syllable sequences
        ser = self._calculate_ser(source_syllables, target_syllables)

        # 2. SCRE (Syllable Count Relative Error) - Average relative error in syllable counts
        scre = self._calculate_scre(source_syllables, target_syllables)

        # 3. ARI (Adjusted Rand Index) - Rhyme clustering agreement
        ari = self._calculate_ari(source_rhyme_scheme, target_rhyme_scheme)

        return EvaluationMetrics(
            ser=ser,
            scre=scre,
            ari=ari,
        )

    def _calculate_ser(
        self,
        source_syllables: list[int],
        target_syllables: list[int],
    ) -> float:
        """
        Calculate Syllable Error Rate using edit distance

        Args:
            source_syllables: Syllable counts for source lines
            target_syllables: Syllable counts for target lines

        Returns:
            SER = EditDistance(source, target) / n
        """
        if not source_syllables:
            return 0.0

        # Calculate Levenshtein distance
        edit_distance = self._levenshtein_distance(source_syllables, target_syllables)
        # Normalize by length
        n = max(len(source_syllables), len(target_syllables))
        return edit_distance / n if n > 0 else 0.0

    def _levenshtein_distance(self, seq1: list, seq2: list) -> int:
        """Calculate Levenshtein distance between two sequences"""
        m, n = len(seq1), len(seq2)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def _calculate_scre(
        self,
        source_syllables: list[int],
        target_syllables: list[int],
    ) -> float:
        """
        Calculate Syllable Count Relative Error

        Args:
            source_syllables: Target syllable counts for each line
            target_syllables: Actual/predicted syllable counts for each line

        Returns:
            SCRE = (1/n) × Σ |target - actual| / target
        """
        if not source_syllables:
            return 0.0

        relative_errors = []
        for src, tgt in zip(source_syllables, target_syllables):
            if src == 0:
                # If target is 0, error is 0 if actual is also 0, else 1.0
                rel_err = 0.0 if tgt == 0 else 1.0
            else:
                rel_err = abs(src - tgt) / src
            relative_errors.append(rel_err)

        return sum(relative_errors) / len(relative_errors) if relative_errors else 0.0

    def _calculate_ari(
        self,
        source_scheme: str,
        target_scheme: str,
    ) -> float:
        """
        Calculate Adjusted Rand Index (ARI) for rhyme clustering agreement

        Measures how well the rhyme clustering (which lines rhyme together)
        is preserved between source and target, independent of label names.

        Args:
            source_scheme: Target rhyme scheme (e.g., "AABB")
            target_scheme: Actual/predicted rhyme scheme

        Returns:
            ARI = [-1, 1] where:
              1.0 = Perfect clustering agreement
              0.0 = Random clustering
             -1.0 = Complete disagreement
        """
        if not source_scheme or not target_scheme:
            return 0.0

        from sklearn.metrics import adjusted_rand_score

        # Convert rhyme schemes to integer labels
        # e.g., "AABB" -> [0, 1, 0, 1]
        source_labels = self._scheme_to_labels(source_scheme)
        target_labels = self._scheme_to_labels(target_scheme)

        # Handle length mismatch: pad shorter one or truncate longer one
        if len(source_labels) != len(target_labels):
            max_len = max(len(source_labels), len(target_labels))
            next_label = max(max(source_labels), max(target_labels)) + 1

            # Pad shorter scheme with unique labels (represents unmatched lines)
            if len(source_labels) < max_len:
                source_labels = source_labels + list(
                    range(next_label, next_label + (max_len - len(source_labels)))
                )
            if len(target_labels) < max_len:
                target_labels = target_labels + list(
                    range(next_label, next_label + (max_len - len(target_labels)))
                )

        # Calculate ARI using sklearn
        return adjusted_rand_score(source_labels, target_labels)

    def _scheme_to_labels(self, scheme: str) -> list[int]:
        """Convert rhyme scheme (e.g., 'AABB') to integer labels (e.g., [0, 1, 0, 1])"""
        label_map = {}
        labels = []
        next_label = 0

        for char in scheme:
            if char not in label_map:
                label_map[char] = next_label
                next_label += 1
            labels.append(label_map[char])

        return labels

