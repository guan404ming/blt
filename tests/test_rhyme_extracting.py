"""
Tests for rhyme extraction functionality in LyricsAnalyzer
"""

import pytest
from blt.translators import LyricsAnalyzer


class TestRhymeExtracting:
    """Test suite for rhyme extraction functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create a LyricsAnalyzer instance for testing"""
        return LyricsAnalyzer()

    @pytest.mark.parametrize(
        "text,lang",
        [
            # English rhyming groups
            ("cat", "en-us"),
            ("bat", "en-us"),
            ("rat", "en-us"),
            ("dog", "en-us"),
            ("log", "en-us"),
            ("day", "en-us"),
            ("way", "en-us"),
            ("night", "en-us"),
            ("light", "en-us"),
            ("hello", "en-us"),
            ("yellow", "en-us"),
            # Chinese words
            ("你", "cmn"),
            ("好", "cmn"),
            ("天", "cmn"),
            # Japanese words
            ("ありがとう", "ja"),
            ("こんにちは", "ja"),
            ("さようなら", "ja"),
        ],
    )
    def test_extract_rhyme_ending_basic(self, analyzer, text, lang):
        """Test rhyme ending extraction returns non-empty string for valid input"""
        result = analyzer.extract_rhyme_ending(text, lang)
        assert len(result) > 0, (
            f"Expected non-empty rhyme ending for '{text}' ({lang}), got '{result}'"
        )

    @pytest.mark.parametrize(
        "words,lang",
        [
            # Simple English rhyming groups (IPA-perfect rhymes)
            (["cat", "bat", "rat"], "en-us"),
            (["day", "way", "say"], "en-us"),
            (["night", "light", "fight"], "en-us"),
            (["tree", "free", "see"], "en-us"),
            (["car", "far", "star"], "en-us"),
            # More complex English rhyming groups
            (["sing", "ring", "bring"], "en-us"),
            (["moon", "soon", "tune"], "en-us"),
            (["great", "straight", "weight"], "en-us"),
            (["phone", "alone", "stone"], "en-us"),
            (["care", "bear", "there"], "en-us"),
            # Multi-syllable rhymes
            (["nation", "station", "creation"], "en-us"),
            (["better", "letter", "getter"], "en-us"),
            (["running", "cunning", "stunning"], "en-us"),
            # Rhymes with different spellings
            (["blue", "flew", "through"], "en-us"),
            (["knight", "night", "right"], "en-us"),
            # Chinese rhyming groups (using pinyin finals/韻母)
            (["天", "年", "連"], "cmn"),  # ian final
            (["家", "夏", "下"], "cmn"),  # ia final
            (["東", "中", "同"], "cmn"),  # ong final
            (["月", "雪", "學"], "cmn"),  # ue final
            (["心", "林", "金"], "cmn"),  # in final
        ],
    )
    def testextract_rhyme_ending_rhyming_groups(self, analyzer, words, lang):
        """Test that rhyming word groups produce the same rhyme ending"""
        endings = [analyzer.extract_rhyme_ending(word, lang) for word in words]
        assert len(set(endings)) == 1, (
            f"Expected all words in {words} to have same rhyme ending, got {endings}"
        )

    @pytest.mark.parametrize(
        "lang",
        ["en-us", "cmn", "ja"],
    )
    def test_extract_rhyme_ending_empty_string(self, analyzer, lang):
        """Test rhyme ending extraction handles empty string"""
        result = analyzer.extract_rhyme_ending("", lang)
        assert result == "", (
            f"Expected empty rhyme ending for empty string ({lang}), got '{result}'"
        )

    def test_extract_rhyme_ending_with_punctuation(self, analyzer):
        """Test rhyme ending extraction handles punctuation"""
        # Test with punctuation
        result_plain = analyzer.extract_rhyme_ending("cat", "en-us")
        result_punct1 = analyzer.extract_rhyme_ending("cat!", "en-us")
        result_punct2 = analyzer.extract_rhyme_ending("cat.", "en-us")

        # All should produce same result
        assert result_plain == result_punct1 == result_punct2, (
            f"Expected same rhyme ending regardless of punctuation, got {result_plain}, {result_punct1}, {result_punct2}"
        )
