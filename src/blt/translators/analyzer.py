"""
Unified Lyrics Analyzer
Centralized core functionality for syllable counting, rhyme detection, and pattern analysis
"""

import os
import re
import hanlp
from pydantic_ai import Agent
from .models import MusicConstraints, WordSegmentation


# Set environment variables for phonemizer
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/opt/homebrew/bin/espeak-ng"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"


class LyricsAnalyzer:
    """
    Unified analyzer for lyrics - handles all core analysis functions

    This class consolidates:
    - Syllable counting (IPA-based)
    - Rhyme detection
    - Syllable pattern analysis
    - Word segmentation
    """

    # IPA patterns
    IPA_VOWEL_PATTERN = r"[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*"
    IPA_DIPHTHONG_PATTERN = r"(?:aɪ|eɪ|ɔɪ|aʊ|oʊ|ɪə|eə|ʊə|aɪə|aʊə|[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*ː?)"

    def __init__(self):
        """Initialize analyzer with lazy-loaded components"""
        self._hanlp_tokenizer = None
        self._segmentation_agent = None

    # ==================== CORE ANALYSIS METHODS ====================

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

    def extract_constraints(
        self, source_lyrics: str, source_lang: str
    ) -> MusicConstraints:
        """
        Extract all music constraints from lyrics

        Args:
            source_lyrics: Source lyrics text
            source_lang: Source language code

        Returns:
            MusicConstraints object
        """
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

    # ==================== PRIVATE HELPERS ====================

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
        """Segment words using LLM (fallback for unsupported languages)"""
        agent = self._get_segmentation_agent()

        lines_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
        prompt = f"""Segment these {language} lyrics lines into words:

{lines_text}

Language: {language}

Return the list of words for each line"""

        try:
            response = agent.run_sync(prompt)
            all_words = response.output.lines
            return [[w for w in words if w.strip()] for words in all_words]
        except Exception as e:
            raise RuntimeError(
                f"LLM word segmentation failed: {e}. "
                "Please ensure GOOGLE_API_KEY is set and valid."
            )

    def _get_hanlp_tokenizer(self):
        """Lazy load HanLP tokenizer"""
        if self._hanlp_tokenizer is None:
            self._hanlp_tokenizer = hanlp.load(
                hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
            )
        return self._hanlp_tokenizer

    def _get_segmentation_agent(self):
        """Lazy load segmentation agent"""
        if self._segmentation_agent is None:
            system_prompt = self._get_segmentation_system_prompt()
            self._segmentation_agent = Agent(
                system_prompt=system_prompt, output_type=WordSegmentation
            )
        return self._segmentation_agent

    def _get_segmentation_system_prompt(self) -> str:
        """Get system prompt for word segmentation"""
        return """You are a word segmentation expert for song lyrics.

Your task is to segment multiple lines of lyrics into individual words or singable units.

CRITICAL RULES:
1. NEVER include punctuation marks as separate words
2. ALWAYS remove all punctuation
3. ONLY return actual words
4. Process ALL lines and return segmentation for each line

For English: Keep contractions together, split on spaces
For Chinese: Split into smallest singable units

Return segmentation for ALL lines, with NO punctuation marks."""
