"""
Tests for LLM-based word segmentation functionality in LyricsAnalyzer
"""

import pytest
from blt.translators import LyricsAnalyzer


class TestWordSegmentation:
    """Test suite for LLM-based word segmentation functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create a LyricsAnalyzer instance for testing"""
        return LyricsAnalyzer(source_lang="English", target_lang="Chinese")

    @pytest.mark.parametrize(
        "text,lang,expected_words",
        [
            # English tests
            ("I don't like you", "en-us", ["I", "don't", "like", "you"]),
            ("Hello world", "en-us", ["Hello", "world"]),
            ("It's a beautiful day", "en-us", ["It's", "a", "beautiful", "day"]),
            ("You can't do that", "en-us", ["You", "can't", "do", "that"]),
            ("Hello, world!", "en-us", ["Hello", "world"]),
            ("I love you.", "en-us", ["I", "love", "you"]),
            # Edge cases
            ("a", "en-us", ["a"]),
            ("I", "en-us", ["I"]),
            # With punctuation
            ("Yes, I can!", "en-us", ["Yes", "I", "can"]),
            ("Don't stop me now", "en-us", ["Don't", "stop", "me", "now"]),
            # Hyphenated words
            ("Well-known", "en-us", ["Well-known"]),
            ("State-of-the-art", "en-us", ["State-of-the-art"]),
            # Multiple spaces
            ("Hello  world", "en-us", ["Hello", "world"]),
            # Chinese tests
            ("我不愛你", "cmn", ["我", "不", "愛", "你"]),
            ("你好世界", "cmn", ["你好", "世界"]),
            ("今天天氣很好", "cmn", ["今天", "天氣", "很", "好"]),
            ("我愛你", "cmn", ["我", "愛", "你"]),
            # Single character Chinese
            ("我", "cmn", ["我"]),
            ("你", "cmn", ["你"]),
        ],
    )
    def test_segment_words(self, analyzer, text, lang, expected_words):
        """Test LLM-based word segmentation for different languages and cases"""
        # _segment_words now takes a list of lines and returns list of lists
        result = analyzer._segment_words([text], lang)
        assert len(result) == 1, f"Expected 1 line result, got {len(result)}"
        assert result[0] == expected_words, (
            f"Expected {expected_words}, got {result[0]}"
        )

    def test_segment_empty_lines(self, analyzer):
        """Test segmentation of empty lines"""
        result = analyzer._segment_words([], "en-us")
        assert result == []

    def test_segment_punctuation_only(self, analyzer):
        """Test segmentation of punctuation-only text"""
        result = analyzer._segment_words(["!!!"], "en-us")
        assert len(result) == 1
        assert result[0] == []

    def test_segment_mixed_punctuation(self, analyzer):
        """Test segmentation with mixed punctuation"""
        result = analyzer._segment_words(["Hello, my name is John!"], "en-us")
        assert len(result) == 1
        assert result[0] == ["Hello", "my", "name", "is", "John"]

    def test_chinese_word_count(self, analyzer):
        """Test Chinese word segmentation count"""
        text = "我愛你"
        result = analyzer._segment_words([text], "cmn")
        assert len(result) == 1
        words = result[0]
        assert len(words) == 3
        assert "".join(words) == "我愛你"

    def test_batch_segment_words(self, analyzer):
        """Test batch segmentation of multiple lines"""
        lines = ["I don't like you", "Hello world", "Yes, I can!"]
        expected = [
            ["I", "don't", "like", "you"],
            ["Hello", "world"],
            ["Yes", "I", "can"],
        ]
        result = analyzer._segment_words(lines, "en-us")
        assert len(result) == 3, f"Expected 3 lines, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"
