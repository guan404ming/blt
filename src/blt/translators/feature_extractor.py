"""
Feature Extractor for Music Constraints
自動提取音樂特徵（音節數、押韻、停頓位置）
"""

import os
import re
from pydantic_ai import Agent
from .models import MusicConstraints
import hanlp

# Set environment variables for phonemizer to find espeak-ng
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/opt/homebrew/bin/espeak-ng"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"


def _get_segmentation_system_prompt() -> str:
    """Get system prompt for word segmentation agent"""
    return """You are a word segmentation expert for song lyrics.

Your task is to segment multiple lines of lyrics into individual words or singable units that match how the lyrics would be sung.

CRITICAL RULES:
1. NEVER include punctuation marks as separate words
2. ALWAYS remove all punctuation (commas, periods, exclamation marks, etc.)
3. ONLY return actual words, not punctuation
4. Process ALL lines and return segmentation for each line

For English and similar languages:
- Keep contractions together (e.g., "don't", "I'm", "you're")
- Keep hyphenated words together (e.g., "twenty-one")
- Split on spaces ONLY
- Remove all punctuation
- Example: "I don't like you." → ["I", "don't", "like", "you"]
- Example: "Yes, I can!" → ["Yes", "I", "can"]

For Chinese and similar languages:
- Split single-character words when appropriate (e.g., "我不愛你" → ["我", "不", "愛", "你"])
- Each character that can stand alone should be separate
- Only combine characters when they form a compound word
- Consider singability - prefer smaller units for song lyrics
- Example: "我不喜欢你" → ["我", "不", "喜欢", "你"]
- Example: "你好世界" → ["你好", "世界"]

Return segmentation for ALL lines, with NO punctuation marks."""


class FeatureExtractor:
    """音樂特徵自動提取器"""

    # IPA vowel pattern including diacritics (combining marks following base vowels)
    # Base vowels + diacritics pattern to handle vowels like o̞, ɯᵝ
    IPA_VOWEL_PATTERN = r"[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*"
    # Common English diphthongs and triphthongs - match these first, then single vowels with optional length marker and diacritics
    IPA_DIPHTHONG_PATTERN = r"(?:aɪ|eɪ|ɔɪ|aʊ|oʊ|ɪə|eə|ʊə|aɪə|aʊə|[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*ː?)"

    def __init__(self, source_lang: str = "English", target_lang: str = "Chinese"):
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Initialize HanLP tokenizer (lazy loaded)
        self._hanlp_tokenizer = None
        self.segmentation_agent = Agent(system_prompt=_get_segmentation_system_prompt())

    def _get_hanlp_tokenizer(self):
        """Lazy load HanLP tokenizer for Chinese word segmentation"""
        if self._hanlp_tokenizer is None:
            self._hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._hanlp_tokenizer

    def extract_constraints(
        self,
        source_lyrics: str,
    ) -> MusicConstraints:
        """
        從源歌詞自動提取音樂約束

        Args:
            source_lyrics: 源語言歌詞

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

        # 3. 詞彙分割 (Word Segmentation) - batch all lines in one LLM call
        syllable_patterns = self._get_syllable_patterns(lines, self.source_lang)

        return MusicConstraints(
            syllable_counts=syllable_counts,
            rhyme_scheme=rhyme_scheme,
            syllable_patterns=syllable_patterns,
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

        使用 phonemizer + espeak-ng 將文本轉換為 IPA，然後計算音節核心數量（元音群組）。
        這個方法支持多語言且更準確。雙元音和長元音被視為單個音節核心。

        Args:
            text: 要計算音節數的文本
            lang: espeak-ng 語言代碼 (例如: 'en-us', 'de', 'fr-fr', 'ja', 'ko', 'cmn')

        Returns:
            音節數量
        """
        # Preprocess: remove punctuation
        punctuation_pattern = r"[,;.!?，。；！？、\s]+"
        cleaned_text = re.sub(punctuation_pattern, "", text)

        if not cleaned_text:
            return 0

        # Chinese: each character is one syllable
        if lang == "cmn":
            return len(cleaned_text)

        # 轉換為 IPA
        ipa_text = self._text_to_ipa(cleaned_text, lang)

        # 計算音節核心（vowel nuclei）
        # 將連續的元音視為一個音節核心（處理雙元音、長元音等）
        syllable_nuclei = re.findall(self.IPA_DIPHTHONG_PATTERN, ipa_text)

        return len(syllable_nuclei)

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
        提取韻腳 (使用 IPA 或拼音)

        將文本轉換為 IPA，然後提取最後的音節作為韻腳。
        對於中文，使用拼音的韻母（final）作為韻腳。

        Args:
            text: 要提取韻腳的文本
            lang: espeak-ng 語言代碼

        Returns:
            韻腳字符串
        """
        text = text.strip()
        if not text:
            return ""

        # For Chinese, use pypinyin to get the final (韻母) of the last character
        if lang == "cmn":
            from pypinyin import pinyin, Style

            # Get the last character's pinyin final (without tone)
            if text:
                # Get pinyin finals for all characters
                finals = pinyin(text, style=Style.FINALS, strict=False)
                if finals and finals[-1]:
                    # Return the final of the last character
                    return finals[-1][0]
            return text

        # 轉換為 IPA for other languages
        ipa_text = self._text_to_ipa(text, lang)

        # 找到所有元音的位置
        vowel_matches = list(re.finditer(self.IPA_VOWEL_PATTERN, ipa_text))

        if not vowel_matches:
            return ""

        # 提取從最後一個元音到字符串結尾的部分作為韻腳
        last_vowel_pos = vowel_matches[-1].start()
        rhyme_ending = ipa_text[last_vowel_pos:]

        return rhyme_ending

    def _get_syllable_patterns(self, lines: list[str], lang: str) -> list[list[int]]:
        """
        分析多行歌詞: 分詞 + 音節計數 (批次處理，一次LLM調用)

        使用 LLM 進行批次分詞，然後計算每個詞的音節數。

        Args:
            lines: 歌詞文本列表 (多行)
            lang: espeak-ng 語言代碼

        Returns:
            每行的音節模式列表，例如 [[1, 1, 3], [1, 2, 1]]

        Example:
            >>> syllables = extractor._get_syllable_patterns(["I like tomato", "You are great"], "en-us")
            >>> print(syllables)   # [[1, 1, 3], [1, 1, 1]]
        """
        # Step 1: Segment all lines using LLM in one call
        all_words = self._segment_words(lines, lang)

        # Step 2: Count syllables for each word in each line
        syllable_patterns = []
        for words in all_words:
            syllables = [self._count_syllables(word, lang) for word in words]
            syllable_patterns.append(syllables)

        return syllable_patterns

    def _segment_words(self, lines: list[str], lang: str) -> list[list[str]]:
        """
        將多行文本分割成詞彙數組 (批次 Word Segmentation)

        使用 LLM 進行智能分詞，一次調用處理所有行。

        Args:
            lines: 歌詞文本列表 (多行)
            lang: espeak-ng 語言代碼

        Returns:
            每行的詞彙列表，例如: [["I", "don't", "like", "you"], ["You", "like", "me"]]

        Raises:
            RuntimeError: 如果 LLM segmentation 失敗且沒有 API key
        """
        if not lines:
            return []

        all_segmented_words = []

        if lang == "cmn":  # Chinese segmentation using HanLP
            tokenizer = self._get_hanlp_tokenizer()
            for line in lines:
                segmented_line = tokenizer(line)
                all_segmented_words.append([word for word in segmented_line if word.strip()])
            return all_segmented_words

        elif lang == "en-us":  # English segmentation using space splitting
            for line in lines:
                # Remove punctuation except for apostrophes and hyphens within words
                cleaned_line = re.sub(r"[^\w\s'-]", "", line)
                # Split by spaces and filter out empty strings
                segmented_line = [word for word in cleaned_line.split() if word.strip()]
                all_segmented_words.append(segmented_line)
            return all_segmented_words

        else:  # Fallback to LLM for other languages
            if not self.segmentation_agent:
                raise RuntimeError(
                    "LLM word segmentation failed. Please ensure GOOGLE_API_KEY is set and valid."
                )

            try:
                # Build prompt for all lines
                lines_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
                prompt = f"""Segment these {lang} lyrics lines into words:

{lines_text}

Language: {lang}

Return the list of words for each line"""

                # Call LLM once for all lines
                response = self.segmentation_agent.run_sync(prompt)
                all_words = response.output.lines

                # Filter empty strings in each line
                return [[w for w in words if w.strip()] for words in all_words]

            except Exception as e:
                # If LLM fails, raise error
                raise RuntimeError(
                    f"LLM word segmentation failed: {e}. Please ensure GOOGLE_API_KEY is set and valid."
                )
