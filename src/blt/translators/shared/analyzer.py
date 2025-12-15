"""
Unified Lyrics Analyzer
Centralized core functionality for syllable counting, rhyme detection, and pattern analysis
"""

from __future__ import annotations

import re
import hanlp
import panphon.distance
from pypinyin import lazy_pinyin
from dotenv import load_dotenv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lyrics_translation.models import MusicConstraints

# Load environment variables from .env file
load_dotenv()

# Initialize panphon feature distance calculator
_ft = panphon.distance.Distance()


class LyricsAnalyzer:
    """
    Unified analyzer for lyrics - handles all core analysis functions

    This class consolidates:
    - Syllable counting (IPA-based)
    - Rhyme detection
    - Syllable pattern analysis
    """

    # IPA patterns
    IPA_VOWEL_PATTERN = r"[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*"
    IPA_DIPHTHONG_PATTERN = r"(?:aɪ|eɪ|ɔɪ|aʊ|oʊ|ɪə|eə|ʊə|aɪə|aʊə|[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*ː?)"

    def __init__(self):
        """Initialize analyzer with lazy-loaded components"""
        self._hanlp_tokenizer = None

    # ==================== CORE ANALYSIS METHODS ====================

    def text_to_ipa(self, text: str, language: str) -> str:
        """
        Convert text to IPA (International Phonetic Alphabet)

        Args:
            text: Text to convert
            language: Language code (e.g., 'en-us', 'cmn', 'ja')

        Returns:
            IPA transcription of the text
        """
        # Remove punctuation for cleaner IPA
        punctuation_pattern = r"[,;.!?，。；！？、]+"
        cleaned_text = re.sub(punctuation_pattern, " ", text).strip()

        if not cleaned_text:
            return ""

        return self._text_to_ipa(cleaned_text, language)

    def count_syllables(self, text: str, language: str) -> int:
        """
        Count syllables in text using IPA-based method

        Args:
            text: Text to analyze
            language: Language code (e.g., 'en-us', 'cmn')

        Returns:
            Number of syllables
        """
        # Remove punctuation
        punctuation_pattern = r"[,;.!?，。；！？、\s]+"
        cleaned_text = re.sub(punctuation_pattern, "", text)

        if not cleaned_text:
            return 0

        # Chinese: each character is one syllable
        if language == "cmn":
            return len(cleaned_text)

        # Convert to IPA and count vowel nuclei
        ipa_text = self._text_to_ipa(cleaned_text, language)
        syllable_nuclei = re.findall(self.IPA_DIPHTHONG_PATTERN, ipa_text)

        return len(syllable_nuclei)

    def extract_rhyme_ending(self, text: str, language: str) -> str:
        """
        Extract rhyme ending from text

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            Rhyme ending string
        """
        text = text.strip()
        if not text:
            return ""

        # For Chinese, use pypinyin to get the final (韻母)
        if language == "cmn":
            from pypinyin import pinyin, Style

            if text:
                finals = pinyin(text, style=Style.FINALS, strict=False)
                if finals and finals[-1]:
                    return finals[-1][0]
            return text

        # For other languages, use IPA
        ipa_text = self._text_to_ipa(text, language)
        vowel_matches = list(re.finditer(self.IPA_VOWEL_PATTERN, ipa_text))

        if not vowel_matches:
            return ""

        last_vowel_pos = vowel_matches[-1].start()
        return ipa_text[last_vowel_pos:]

    def check_rhyme(self, text1: str, text2: str, language: str) -> bool:
        """
        Check if two texts rhyme

        Args:
            text1: First text
            text2: Second text
            language: Language code

        Returns:
            True if texts rhyme
        """
        rhyme1 = self.extract_rhyme_ending(text1, language)
        rhyme2 = self.extract_rhyme_ending(text2, language)

        if not rhyme1 or not rhyme2:
            return False

        return rhyme1 == rhyme2 or rhyme1 in rhyme2 or rhyme2 in rhyme1

    def get_syllable_patterns(self, lines: list[str], language: str) -> list[list[int]]:
        """
        Get syllable pattern (syllables per word) for multiple lines

        Args:
            lines: List of text lines
            language: Language code

        Returns:
            List of syllable patterns, e.g., [[1, 1, 3], [1, 2, 1]]
        """
        # Segment words for all lines
        all_words = self._segment_words(lines, language)

        # Count syllables for each word
        syllable_patterns = []
        for words in all_words:
            syllables = [self.count_syllables(word, language) for word in words]
            syllable_patterns.append(syllables)

        return syllable_patterns

    def detect_rhyme_scheme(self, lines: list[str], language: str) -> str:
        """
        Detect rhyme scheme from lines

        Args:
            lines: List of text lines
            language: Language code

        Returns:
            Rhyme scheme string (e.g., "AABB")
        """
        if len(lines) < 2:
            return "A"

        # Extract rhyme endings
        rhyme_endings = [self.extract_rhyme_ending(line, language) for line in lines]

        # Build rhyme scheme
        scheme = []
        rhyme_map = {}
        current_label = ord("A")

        for ending in rhyme_endings:
            if ending in rhyme_map:
                scheme.append(rhyme_map[ending])
            else:
                label = chr(current_label)
                rhyme_map[ending] = label
                scheme.append(label)
                current_label += 1

        return "".join(scheme)

    def extract_constraints(self, source_lyrics: str, source_lang: str):
        """
        Extract all music constraints from lyrics

        Args:
            source_lyrics: Source lyrics text
            source_lang: Source language code

        Returns:
            MusicConstraints object
        """
        from ..lyrics_translation.models import MusicConstraints

        lines = [
            line.strip() for line in source_lyrics.strip().split("\n") if line.strip()
        ]

        syllable_counts = [self.count_syllables(line, source_lang) for line in lines]
        rhyme_scheme = self.detect_rhyme_scheme(lines, source_lang)
        syllable_patterns = self.get_syllable_patterns(lines, source_lang)

        return MusicConstraints(
            syllable_counts=syllable_counts,
            rhyme_scheme=rhyme_scheme,
            syllable_patterns=syllable_patterns,
        )

    def calculate_ipa_similarity(
        self, ipa1: str, ipa2: str, is_chinese: bool = False
    ) -> float:
        """
        Calculate phonetic similarity between two IPA strings using panphon feature-based distance

        Args:
            ipa1: First IPA string (or Chinese text if is_chinese=True)
            ipa2: Second IPA string (or Chinese text if is_chinese=True)
            is_chinese: If True, convert Chinese to pinyin first

        Returns:
            Similarity score between 0 and 1
        """
        if not ipa1 and not ipa2:
            return 1.0
        if not ipa1 or not ipa2:
            return 0.0

        # For Chinese, convert to pinyin first
        if is_chinese:
            ipa1 = self._chinese_to_pinyin(ipa1)
            ipa2 = self._chinese_to_pinyin(ipa2)

        # Normalize: remove spaces, convert to lowercase
        ipa1 = ipa1.replace(" ", "").lower()
        ipa2 = ipa2.replace(" ", "").lower()

        len1, len2 = len(ipa1), len(ipa2)

        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0

        # Calculate feature-based edit distance using panphon
        distance = _ft.feature_edit_distance(ipa1, ipa2)
        max_len = max(len1, len2)

        # Convert distance to similarity score
        similarity = 1.0 - (distance / max_len)

        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    def analyze_pattern_alignment(
        self, target_pattern: list[int], current_pattern: list[int]
    ) -> dict:
        """
        Analyze alignment between target and current syllable patterns.
        Returns detailed feedback on mismatches and suggestions.

        Args:
            target_pattern: Target syllable pattern, e.g., [1, 2, 2, 1]
            current_pattern: Current/actual syllable pattern, e.g., [1, 1, 2, 2]

        Returns:
            Dictionary containing:
                - 'matches': bool - Whether patterns match exactly
                - 'similarity': float - Similarity score (0-1)
                - 'differences': list[dict] - Word-by-word differences
                - 'suggestions': list[str] - Improvement suggestions
                - 'total_syllables_match': bool - Whether total syllable count matches
        """
        if not target_pattern or not current_pattern:
            return {
                "matches": target_pattern == current_pattern,
                "similarity": 0.0 if target_pattern != current_pattern else 1.0,
                "differences": [],
                "suggestions": [],
                "total_syllables_match": sum(target_pattern or [])
                == sum(current_pattern or []),
            }

        target_total = sum(target_pattern)
        current_total = sum(current_pattern)
        exact_match = target_pattern == current_pattern

        # Calculate position-by-position differences
        differences = []
        max_len = max(len(target_pattern), len(current_pattern))

        for i in range(max_len):
            target_val = target_pattern[i] if i < len(target_pattern) else 0
            current_val = current_pattern[i] if i < len(current_pattern) else 0
            diff = current_val - target_val

            if diff != 0:
                differences.append(
                    {
                        "word_position": i,
                        "target_syllables": target_val,
                        "current_syllables": current_val,
                        "difference": diff,
                    }
                )

        # Generate suggestions
        suggestions = []
        if not exact_match:
            if len(target_pattern) != len(current_pattern):
                suggestions.append(
                    f"Word count mismatch: target has {len(target_pattern)} words, current has {len(current_pattern)} words"
                )

            if target_total != current_total:
                syllable_diff = target_total - current_total
                if syllable_diff > 0:
                    suggestions.append(f"Need to add {syllable_diff} syllables overall")
                else:
                    suggestions.append(
                        f"Need to remove {-syllable_diff} syllables overall"
                    )

            for diff_info in differences:
                pos = diff_info["word_position"]
                diff = diff_info["difference"]
                if diff > 0:
                    suggestions.append(
                        f"Word {pos + 1}: has {diff} too many syllables (target {diff_info['target_syllables']}, current {diff_info['current_syllables']})"
                    )
                else:
                    suggestions.append(
                        f"Word {pos + 1}: needs {-diff} more syllables (target {diff_info['target_syllables']}, current {diff_info['current_syllables']})"
                    )

        # Calculate similarity score using simple metrics
        # 1. Position-by-position differences
        position_similarity = (
            1.0
            if not differences
            else 1.0 - (sum(abs(d["difference"]) for d in differences) / target_total)
        )

        # 2. Length difference penalty
        length_similarity = (
            1.0
            if len(target_pattern) == len(current_pattern)
            else max(
                0.0,
                1.0 - (abs(len(target_pattern) - len(current_pattern)) / max_len),
            )
        )

        # 3. Total syllable match penalty
        total_similarity = (
            1.0
            if target_total == current_total
            else max(0.0, 1.0 - (abs(target_total - current_total) / target_total))
        )

        # Weighted similarity (prioritize exact matches in key areas)
        similarity = (
            (position_similarity * 0.5)
            + (length_similarity * 0.25)
            + (total_similarity * 0.25)
        )

        return {
            "matches": exact_match,
            "similarity": max(0.0, min(1.0, similarity)),
            "differences": differences,
            "suggestions": suggestions,
            "total_syllables_match": target_total == current_total,
        }

    def score_syllable_patterns(
        self, target_patterns: list[list[int]], current_patterns: list[list[int]]
    ) -> dict:
        """
        Score overall syllable pattern quality across all lines.
        Provides comprehensive metrics for pattern matching.

        Args:
            target_patterns: Target syllable patterns for all lines
            current_patterns: Current syllable patterns for all lines

        Returns:
            Dictionary containing:
                - 'overall_score': float (0-1) - Weighted overall score
                - 'exact_match_rate': float (0-1) - Percentage of exact matches
                - 'fuzzy_match_rate': float (0-1) - Percentage of fuzzy matches (>0.8 similarity)
                - 'average_similarity': float (0-1) - Average similarity across all lines
                - 'worst_line': dict - Info about the worst-matching line
                - 'best_line': dict - Info about the best-matching line
                - 'total_syllables_error': int - Absolute error in total syllables
                - 'pattern_distribution_score': float - How well word distribution matches
        """
        if not target_patterns or not current_patterns:
            return {
                "overall_score": 0.0,
                "exact_match_rate": 0.0,
                "fuzzy_match_rate": 0.0,
                "average_similarity": 0.0,
                "worst_line": None,
                "best_line": None,
                "total_syllables_error": 0,
                "pattern_distribution_score": 0.0,
            }

        # Align patterns (handle different lengths)
        max_lines = max(len(target_patterns), len(current_patterns))
        alignments = []
        exact_matches = 0
        fuzzy_matches = 0  # 0.8+ similarity
        total_similarity = 0.0
        worst_alignment = None
        worst_similarity = 1.0
        best_alignment = None
        best_similarity = 0.0

        for i in range(max_lines):
            target = target_patterns[i] if i < len(target_patterns) else []
            current = current_patterns[i] if i < len(current_patterns) else []

            alignment = self.analyze_pattern_alignment(target, current)
            alignments.append((i, alignment))

            # Track statistics
            similarity = alignment["similarity"]
            total_similarity += similarity

            if alignment["matches"]:
                exact_matches += 1

            if similarity >= 0.8:
                fuzzy_matches += 1

            # Track worst and best
            if similarity < worst_similarity:
                worst_similarity = similarity
                worst_alignment = (i, alignment)

            if similarity > best_similarity:
                best_similarity = similarity
                best_alignment = (i, alignment)

        # Calculate aggregate scores
        exact_match_rate = exact_matches / max_lines if max_lines > 0 else 0.0
        fuzzy_match_rate = fuzzy_matches / max_lines if max_lines > 0 else 0.0
        average_similarity = total_similarity / max_lines if max_lines > 0 else 0.0

        # Calculate total syllables error
        target_total = sum(sum(p) for p in target_patterns)
        current_total = sum(sum(p) for p in current_patterns)
        total_syllables_error = abs(target_total - current_total)

        # Pattern distribution score: balance between exact matches and fuzzy matches
        # Heavily weight exact matches, moderately weight fuzzy matches
        pattern_distribution_score = (exact_match_rate * 0.7) + (fuzzy_match_rate * 0.3)

        # Overall score: combination of exact matches, average similarity, and total syllables
        syllable_score = (
            1.0
            if total_syllables_error == 0
            else max(0.0, 1.0 - (total_syllables_error / target_total))
        )
        overall_score = (
            (exact_match_rate * 0.4)
            + (average_similarity * 0.3)
            + (syllable_score * 0.3)
        )

        return {
            "overall_score": max(0.0, min(1.0, overall_score)),
            "exact_match_rate": exact_match_rate,
            "fuzzy_match_rate": fuzzy_match_rate,
            "average_similarity": average_similarity,
            "worst_line": {
                "line_idx": worst_alignment[0],
                "similarity": worst_alignment[1]["similarity"],
                "target": worst_alignment[1].get("target_pattern", []),
                "current": worst_alignment[1].get("current_pattern", []),
            }
            if worst_alignment
            else None,
            "best_line": {
                "line_idx": best_alignment[0],
                "similarity": best_alignment[1]["similarity"],
                "target": best_alignment[1].get("target_pattern", []),
                "current": best_alignment[1].get("current_pattern", []),
            }
            if best_alignment
            else None,
            "total_syllables_error": total_syllables_error,
            "pattern_distribution_score": pattern_distribution_score,
        }

    # ==================== PRIVATE HELPERS ====================

    def _chinese_to_pinyin(self, text: str) -> str:
        """Convert Chinese text to pinyin"""
        # Convert Chinese to pinyin
        pinyin_list = lazy_pinyin(text)
        pinyin_text = " ".join(pinyin_list)
        return pinyin_text

    def _text_to_ipa(self, text: str, lang: str) -> str:
        """Convert text to IPA using phonemizer"""
        from phonemizer import phonemize

        return phonemize(text, language=lang, backend="espeak", strip=True)

    def _segment_words(self, lines: list[str], language: str) -> list[list[str]]:
        """
        Segment lines into words

        Args:
            lines: List of text lines
            language: Language code

        Returns:
            List of word lists for each line
        """
        if not lines:
            return []

        all_segmented_words = []

        if language == "cmn":
            # Chinese segmentation using HanLP
            tokenizer = self._get_hanlp_tokenizer()
            for line in lines:
                segmented_line = tokenizer(line)
                all_segmented_words.append(
                    [word for word in segmented_line if word.strip()]
                )
            return all_segmented_words

        elif language == "en-us":
            # English segmentation using space splitting
            for line in lines:
                cleaned_line = re.sub(r"[^\w\s'-]", "", line)
                segmented_line = [word for word in cleaned_line.split() if word.strip()]
                all_segmented_words.append(segmented_line)
            return all_segmented_words

        else:
            # Fallback to LLM for other languages
            return self._segment_with_llm(lines, language)

    def _segment_with_llm(self, lines: list[str], language: str) -> list[list[str]]:
        """Segment words using simple space splitting (fallback for unsupported languages)"""
        # For unsupported languages, use simple space-based splitting
        # This is a fallback and may not be accurate for all languages
        all_segmented_words = []
        for line in lines:
            # Remove punctuation and split on whitespace
            cleaned_line = re.sub(r"[^\w\s'-]", "", line)
            segmented_line = [word for word in cleaned_line.split() if word.strip()]
            all_segmented_words.append(segmented_line)
        return all_segmented_words

    def _get_hanlp_tokenizer(self):
        """Lazy load HanLP tokenizer"""
        if self._hanlp_tokenizer is None:
            self._hanlp_tokenizer = hanlp.load(
                hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
            )
        return self._hanlp_tokenizer
