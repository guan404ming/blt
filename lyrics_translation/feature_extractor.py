"""
Feature Extractor for Music Constraints
自動提取音樂特徵（音節數、押韻、停頓位置）
"""

import re
from typing import Optional
from .models import MusicConstraints


class FeatureExtractor:
    """音樂特徵自動提取器"""

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

    def _count_syllables(self, text: str, lang: str) -> int:
        """計算文本的音節數"""
        if lang.lower() in ["chinese", "mandarin", "zh", "中文"]:
            # 中文: 字符數 + 拉長音符號數量
            chinese_chars = len(re.sub(r"[^\u4e00-\u9fff]", "", text))
            # 計算拉長音符號 '-' 的數量
            elongation_marks = text.count("-")
            return chinese_chars + elongation_marks

        elif lang.lower() in ["english", "en"]:
            # 英文: 使用 CMU Pronouncing Dictionary
            try:
                import pronouncing

                # 分詞並計算每個詞的音節數
                # 使用更好的正則表達式來保留縮寫詞（contractions）
                words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
                total_syllables = 0

                for word in words:
                    # 獲取發音列表
                    phones = pronouncing.phones_for_word(word)

                    if phones:
                        # 使用第一個發音，計算音節數（音節 = 重音標記數）
                        syllable_count = pronouncing.syllable_count(phones[0])
                        total_syllables += syllable_count
                    else:
                        # 如果字典中沒有，使用簡單的元音計數法
                        total_syllables += self._fallback_syllable_count(word)

                return total_syllables

            except ImportError:
                # Fallback: 簡單的元音計數
                words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
                return sum(self._fallback_syllable_count(word) for word in words)

        elif lang.lower() in ["japanese", "ja", "日文"]:
            return self._count_syllables_japanese(text)

        else:
            return self._count_syllables_other(text)

    def _fallback_syllable_count(self, word: str) -> int:
        """簡單的元音計數法（備用）"""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel

        # 調整: 以 'e' 結尾通常不發音（除非是單音節詞）
        if word.endswith("e") and count > 1:
            count -= 1

        # 調整: 以 'le' 結尾通常發音
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)

    def _count_syllables_japanese(self, text: str) -> int:
        """日文音節計數"""
        # 日文: 假名數
        # 簡化版: 統計字符數
        return len(re.sub(r"[^\u3040-\u309f\u30a0-\u30ff]", "", text))

    def _count_syllables_other(self, text: str) -> int:
        """其他語言音節計數"""
        # 其他語言: 使用空格分詞估算
        words = text.split()
        return len(words) * 2  # 粗略估計

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
        """提取韻腳（末字或末音節）"""
        text = text.strip()

        if lang.lower() in ["chinese", "mandarin", "zh", "中文"]:
            # 中文: 提取末字的韻母
            try:
                from pypinyin import lazy_pinyin, Style

                words = re.findall(r"[\u4e00-\u9fff]", text)
                if words:
                    last_char = words[-1]
                    pinyin = lazy_pinyin(last_char, style=Style.FINALS)
                    return pinyin[0] if pinyin else last_char
            except ImportError:
                # Fallback: 直接返回末字
                words = re.findall(r"[\u4e00-\u9fff]", text)
                return words[-1] if words else ""

        elif lang.lower() in ["english", "en"]:
            # 英文: 提取末詞的韻母
            # 使用更好的正則表達式來保留縮寫詞（contractions）
            words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
            if words:
                last_word = words[-1]
                # 簡單的韻腳檢測: 最後2-3個字母
                return last_word[-3:] if len(last_word) >= 3 else last_word
            return ""

        else:
            # 其他語言: 返回末詞
            words = text.split()
            return words[-1] if words else ""

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
