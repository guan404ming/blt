"""
Tests for syllable counting functionality in FeatureExtractor
"""

import pytest
from blt.translators.feature_extractor import FeatureExtractor


class TestSyllableCounting:
    """Test suite for syllable counting functionality"""

    @pytest.fixture
    def extractor(self):
        """Create a FeatureExtractor instance for testing"""
        return FeatureExtractor(source_lang="English", target_lang="Chinese")

    @pytest.mark.parametrize(
        "text,lang,expected_count",
        [
            # English tests (actual syllable counting)
            # Diphthongs are now correctly counted as single syllables
            ("hello", "en-us", 2),  # hel-lo: həloʊ = 2 syllables
            ("world", "en-us", 1),  # world: wɜːld = 1 syllable
            ("beautiful", "en-us", 3),  # beau-ti-ful: 3 syllables
            ("cat", "en-us", 1),  # cat: kæt = 1 syllable
            ("dictionary", "en-us", 4),  # dic-tion-ar-y: 4 syllables
            ("amazing", "en-us", 3),  # a-maz-ing: ɐmeɪzɪŋ = 3 syllables
            ("the", "en-us", 1),  # the: ðə = 1 syllable
            ("computer", "en-us", 3),  # com-pu-ter: kəmpjuːɾɚ = 3 syllables
            ("programming", "en-us", 3),  # pro-gram-ming: pɹoʊɡɹæmɪŋ = 3 syllables
            ("syllable", "en-us", 3),  # syl-la-ble: 3 syllables
            # Multi-word English phrases
            ("hello world", "en-us", 3),  # hel-lo world = 3 syllables
            ("how are you", "en-us", 3),  # how are you = 3 syllables
            # Edge cases English
            ("", "en-us", 0),
            ("a", "en-us", 1),  # eɪ = 1 syllable (diphthong)
            ("I", "en-us", 1),  # aɪ = 1 syllable (diphthong)
            # Chinese tests - each character is typically one syllable
            ("你好", "cmn", 2),
            ("世界", "cmn", 2),
            ("音樂", "cmn", 2),
            ("翻譯", "cmn", 2),
            ("中國", "cmn", 2),
            ("一二三", "cmn", 3),
            ("美好的一天", "cmn", 5),
            # Single characters Chinese
            ("我", "cmn", 1),
            ("你", "cmn", 1),
            ("是", "cmn", 1),
            # Edge cases Chinese
            ("", "cmn", 0),
            # Longer phrases Chinese
            ("今天天氣很好", "cmn", 6),
            # Japanese tests (hiragana works better than kanji for IPA)
            ("こんにちは", "ja", 4),  # ko̞nnitɕihä
            ("ありがとう", "ja", 5),  # äɽiɡäto̞ɯᵝ
            ("さようなら", "ja", 5),  # säjo̞ɯᵝnäɽä
            ("おはよう", "ja", 4),  # o̞häjo̞ɯᵝ
            # Edge cases Japanese
            ("", "ja", 0),
        ],
    )
    def test_count_syllables_basic(self, extractor, text, lang, expected_count):
        """Test syllable counting for English, Chinese, and Japanese with basic words and phrases"""
        result = extractor._count_syllables(text, lang)
        assert result == expected_count, (
            f"Expected {expected_count} syllables for '{text}' ({lang}), got {result}"
        )

    @pytest.mark.parametrize(
        "text,lang,expected_count",
        [
            # Longer English sentences
            (
                "The quick brown fox jumps over the lazy dog",
                "en-us",
                11,  # The(1) quick(1) brown(1) fox(1) jumps(1) o-ver(2) the(1) la-zy(2) dog(1)
            ),
            (
                "Artificial intelligence is transforming the world",
                "en-us",
                14,  # Ar-ti-fi-cial(4) in-tel-li-gence(4) is(1) trans-form-ing(3) the(1) world(1)
            ),
            (
                "She sells seashells by the seashore",
                "en-us",
                8,  # She(1) sells(1) sea-shells(2) by(1) the(1) sea-shore(2)
            ),
            (
                "Peter Piper picked a peck of pickled peppers",
                "en-us",
                12,  # Pe-ter(2) Pi-per(2) picked(1) a(1) peck(1) of(1) pick-led(2) pep-pers(2)
            ),
            # Longer Chinese sentences
            (
                "春眠不覺曉，處處聞啼鳥",
                "cmn",
                10,  # 10 chars (punctuation removed)
            ),
            (
                "人工智能正在改變世界",
                "cmn",
                10,
            ),
            (
                "中華人民共和國是一個偉大的國家",
                "cmn",
                15,
            ),
            (
                "我們在這裡學習中文歌曲翻譯",
                "cmn",
                13,
            ),
            # Longer Japanese sentences (using hiragana)
            (
                "きょうはとてもいいてんきです",
                "ja",
                8,  # IPA-based count
            ),
        ],
    )
    def test_count_syllables_longer_sentences(
        self, extractor, text, lang, expected_count
    ):
        """Test syllable counting for longer sentences in English, Chinese, and Japanese"""
        result = extractor._count_syllables(text, lang)
        assert result == expected_count, (
            f"Expected {expected_count} syllables for '{text}' ({lang}), got {result}"
        )

    @pytest.mark.parametrize(
        "text,lang",
        [
            # English with punctuation
            ("hello, world!", "en-us"),
            ("What's up?", "en-us"),
            # Chinese with punctuation
            ("你好！", "cmn"),
            # Japanese with punctuation
            ("こんにちは！", "ja"),
        ],
    )
    def test_count_syllables_with_punctuation(self, extractor, text, lang):
        """Test that syllable counting handles punctuation correctly for all languages"""
        result = extractor._count_syllables(text, lang)
        # Should still return a positive count even with punctuation
        assert result > 0, (
            f"Expected positive syllable count for '{text}' ({lang}), got {result}"
        )

    @pytest.mark.parametrize(
        "lang",
        ["en-us", "cmn", "ja"],
    )
    def test_count_syllables_whitespace(self, extractor, lang):
        """Test syllable counting with whitespace-only input for all languages"""
        result = extractor._count_syllables("   ", lang)
        assert result == 0, (
            f"Expected 0 syllables for whitespace ({lang}), got {result}"
        )
