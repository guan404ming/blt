"""
Feature Extractor for Music Constraints
自動提取音樂特徵（音節數、押韻、停頓位置）
"""

import os
import re
from typing import Optional
from .models import MusicConstraints

# Set environment variables for phonemizer to find espeak-ng
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/opt/homebrew/bin/espeak-ng"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"


class FeatureExtractor:
    """音樂特徵自動提取器"""

    IPA_VOWEL_PATTERN = r"[iɪeɛæaɑɒɔoʊuʉɨəɜɞʌyøœɶɐ]"

    def __init__(self, source_lang: str = "English", target_lang: str = "Chinese"):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def extract_constraints(
        self, source_lyrics: str, music_file: Optional[str] = None
    ) -> MusicConstraints:
        """
        從源歌詞自動提取音樂約束

        Args:
            source_lyrics: 源語言歌詞
            music_file: 可選的 MIDI/MusicXML 檔案路徑

        Returns:
            MusicConstraints: 音樂約束條件
        """
        lines = [
            line.strip() for line in source_lyrics.strip().split("\n") if line.strip()
        ]

        # 1. 音節數計算
        syllable_counts = [
            self._count_syllables(line, self.source_lang) for line in lines
        ]

        # 2. 押韻方案檢測
        rhyme_scheme = self._detect_rhyme_scheme(lines, self.source_lang)

        # 3. 停頓位置推測 (如果沒有音樂檔案，基於標點符號推測)
        if music_file:
            pause_positions = self._extract_pauses_from_music(music_file)
        else:
            pause_positions = self._infer_pauses(lines)

        return MusicConstraints(
            syllable_counts=syllable_counts,
            rhyme_scheme=rhyme_scheme,
            pause_positions=pause_positions,
        )

    def _text_to_ipa(self, text: str, lang: str) -> str:
        """
        將文本轉換為 IPA (國際音標)

        使用 phonemizer + espeak-ng 進行多語言文本到 IPA 的轉換。

        Args:
            text: 要轉換的文本
            lang: espeak-ng 語言代碼 (例如: 'en-us', 'de', 'fr-fr', 'ja', 'ko', 'cmn')

        Returns:
            IPA 格式的音標字符串
        """
        from phonemizer import phonemize

        ipa_text = phonemize(text, language=lang, backend="espeak", strip=True)
        return ipa_text

    def _count_syllables(self, text: str, lang: str) -> int:
        """
        計算文本的音節數 (使用 IPA-based 方法)

        使用 phonemizer + espeak-ng 將文本轉換為 IPA，然後計算元音數量來估計音節數。
        這個方法支持多語言且更準確。

        Args:
            text: 要計算音節數的文本
            lang: espeak-ng 語言代碼 (例如: 'en-us', 'de', 'fr-fr', 'ja', 'ko', 'cmn')

        Returns:
            音節數量
        """
        # 轉換為 IPA
        ipa_text = self._text_to_ipa(text, lang)
        vowels = re.findall(self.IPA_VOWEL_PATTERN, ipa_text)

        # 計算 IPA 中的元音數量（元音 = 音節核心）
        return len(vowels) if lang != "cmn" else len(text)

    def _detect_rhyme_scheme(self, lines: list[str], lang: str) -> str:
        """檢測押韻方案"""
        if len(lines) < 2:
            return "A"

        # 提取每行的韻腳
        rhyme_endings = []
        for line in lines:
            ending = self._extract_rhyme_ending(line, lang)
            rhyme_endings.append(ending)

        # 分析押韻模式
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

    def _extract_rhyme_ending(self, text: str, lang: str) -> str:
        """
        提取韻腳 (使用 IPA)

        將文本轉換為 IPA，然後提取最後的音節作為韻腳。
        韻腳包含最後一個元音及其後的所有輔音。

        Args:
            text: 要提取韻腳的文本
            lang: espeak-ng 語言代碼

        Returns:
            IPA 格式的韻腳
        """
        text = text.strip()
        if not text:
            return ""

        # 轉換為 IPA
        ipa_text = self._text_to_ipa(text, lang)

        # 找到所有元音的位置
        vowel_matches = list(re.finditer(self.IPA_VOWEL_PATTERN, ipa_text))

        if not vowel_matches:
            return ""

        # 提取從最後一個元音到字符串結尾的部分作為韻腳
        last_vowel_pos = vowel_matches[-1].start()
        rhyme_ending = ipa_text[last_vowel_pos:]

        return rhyme_ending

    def _infer_pauses(self, lines: list[str]) -> list[int]:
        """基於標點符號推測停頓位置"""
        pause_positions = []
        cumulative_syllables = 0

        for line in lines:
            # 檢測標點符號位置
            punctuation = r"[,;.!?，。；！？、]"
            parts = re.split(punctuation, line)

            for i, part in enumerate(parts[:-1]):  # 不包括最後一個
                syllables = self._count_syllables(part, self.source_lang)
                cumulative_syllables += syllables
                pause_positions.append(cumulative_syllables)

            # 行尾也是停頓
            cumulative_syllables += self._count_syllables(parts[-1], self.source_lang)
            pause_positions.append(cumulative_syllables)

        return pause_positions

    def _extract_pauses_from_music(self, music_file: str) -> list[int]:
        """從 MIDI/MusicXML 提取停頓位置"""
        # TODO: 實作音樂檔案解析
        # 需要使用 music21 或類似的函式庫
        raise NotImplementedError("Music file parsing not implemented yet")
