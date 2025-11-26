"""
Lyrics Translator using PydanticAI + Gemini 2.0 Flash
核心翻譯器實作
"""

import os
from datetime import datetime
from typing import Optional
from pydantic_ai import Agent

from .models import LyricTranslation, MusicConstraints
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator


class LyricsTranslator:
    """歌詞翻譯器"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        auto_save: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        初始化翻譯器

        Args:
            model: Gemini 模型名稱
            api_key: Google AI API Key (若未提供則從環境變數讀取)
            auto_save: 是否自動保存翻譯結果
            save_dir: 保存目錄（若未提供則使用 'outputs'）
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")

        # Set API key in environment for pydantic-ai
        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.auto_save = auto_save
        self.save_dir = save_dir or "outputs"

        # 工具
        self.feature_extractor = FeatureExtractor()
        self.validator = ConstraintValidator()

        # 初始化 Agent - pydantic-ai will infer Google provider from model name
        self.agent = Agent(
            model=model,
            output_type=LyricTranslation,
            system_prompt=self._get_system_prompt(),
        )

        # 註冊驗證工具供 LLM 調用
        self._register_validation_tools()

    def _register_validation_tools(self):
        """註冊驗證工具供 LLM 在生成過程中使用"""

        def count_syllables_impl(text: str, language: str) -> int:
            """
            計算文本的音節數

            Args:
                text: 要計算音節數的文本
                language: 語言代碼 (例如: 'en-us', 'cmn', 'ja', 'ko')

            Returns:
                音節數量
            """
            result = self.feature_extractor._count_syllables(text, language)
            print(f"[TOOL] count_syllables('{text}', '{language}') = {result}")
            return result

        def check_rhyme_impl(text1: str, text2: str, language: str) -> dict:
            """
            檢查兩個文本是否押韻

            Args:
                text1: 第一個文本
                text2: 第二個文本
                language: 語言代碼 (例如: 'en-us', 'cmn', 'ja', 'ko')

            Returns:
                包含押韻檢查結果的字典:
                - rhymes: 是否押韻 (bool)
                - rhyme1: 第一個文本的韻腳
                - rhyme2: 第二個文本的韻腳
            """
            rhyme1 = self.feature_extractor._extract_rhyme_ending(text1, language)
            rhyme2 = self.feature_extractor._extract_rhyme_ending(text2, language)

            # 判斷是否押韻
            rhymes = bool(
                rhyme1
                and rhyme2
                and (rhyme1 == rhyme2 or rhyme1 in rhyme2 or rhyme2 in rhyme1)
            )

            result = {"rhymes": rhymes, "rhyme1": rhyme1, "rhyme2": rhyme2}
            print(f"[TOOL] check_rhyme('{text1}', '{text2}', '{language}') = {result}")
            return result

        def get_rhyme_ending_impl(text: str, language: str) -> str:
            """
            提取文本的韻腳

            Args:
                text: 要提取韻腳的文本
                language: 語言代碼 (例如: 'en-us', 'cmn', 'ja', 'ko')

            Returns:
                韻腳字符串
            """
            result = self.feature_extractor._extract_rhyme_ending(text, language)
            print(f"[TOOL] get_rhyme_ending('{text}', '{language}') = '{result}'")
            return result

        # 註冊工具
        self.agent.tool_plain(count_syllables_impl)
        self.agent.tool_plain(check_rhyme_impl)
        self.agent.tool_plain(get_rhyme_ending_impl)

    def _get_system_prompt(self) -> str:
        """獲取系統 prompt"""
        return """你是專業的歌詞翻譯專家。

你可以使用以下工具來驗證翻譯品質：
- count_syllables(text, language): 計算文本的音節數
- check_rhyme(text1, text2, language): 檢查兩個文本是否押韻
- get_rhyme_ending(text, language): 提取文本的韻腳

約束優先級：
⭐⭐⭐ 音節數（絕對必須符合，不可妥協）- 這是最重要的約束
⭐⭐ 押韻（盡量滿足，可以適度放寬）
⭐ 停頓位置（參考即可）

請將歌詞翻譯成目標語言，並遵守以下要求:
1. **【絕對必須】嚴格遵守音節數限制 - 必須使用 count_syllables 工具驗證每一行，音節數必須完全符合**
2. 保持原意和情感
3. 符合目標語言的自然表達
4. 在指定位置押韻 - 使用 check_rhyme 工具驗證押韻（可以適度放寬）
5. 避免在音樂停頓處斷詞

工作流程：
1. 草擬每一行的翻譯
2. **【最重要】使用 count_syllables 驗證音節數是否完全符合要求**
3. **如果音節數不符，必須調整翻譯並重新驗證，直到完全符合為止**
4. 對需要押韻的行，使用 check_rhyme 驗證押韻（盡力而為，但音節數優先）
5. 如果押韻不符但音節數正確，可以接受
6. 確保所有行的音節數都完全符合後，輸出最終結果

請以結構化格式輸出翻譯結果。音節數的準確性是評估翻譯品質的最重要指標。"""

    def translate(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        constraints: Optional[MusicConstraints] = None,
        save_path: Optional[str] = None,
        save_format: str = "json",
    ) -> LyricTranslation:
        """
        翻譯歌詞

        Args:
            source_lyrics: 源語言歌詞
            source_lang: 源語言
            target_lang: 目標語言
            constraints: 音樂約束（若未提供則自動提取）
            auto_retry: 約束不滿足時是否自動重試
            save_path: 保存路徑（覆蓋 auto_save 設定）
            save_format: 保存格式 ("json", "txt", "md")

        Returns:
            LyricTranslation: 翻譯結果
        """
        # 1. 提取約束（如果未提供）
        if constraints is None:
            self.feature_extractor.source_lang = source_lang
            self.feature_extractor.target_lang = target_lang
            constraints = self.feature_extractor.extract_constraints(source_lyrics)

        # 2. 構建 prompt
        user_prompt = self._build_prompt(
            source_lyrics=source_lyrics,
            source_lang=source_lang,
            target_lang=target_lang,
            constraints=constraints,
        )

        # 3. 呼叫 LLM
        result = self.agent.run_sync(user_prompt)
        translation = result.output

        # 4. 重新計算翻譯結果的實際音節數
        translation = self._recalculate_syllables(translation, target_lang)

        # 5. 最終驗證並顯示結果（不進行自動重試，信任 LLM 的自我驗證）
        self.validator.target_lang = target_lang
        validation_result = self.validator.validate(translation, constraints)

        if validation_result.passed:
            print("✓ 所有約束都已滿足")
        else:
            print(f"⚠ 約束滿足度: {validation_result.score:.2%}")

        # 6. 保存結果（如果啟用）
        if save_path or self.auto_save:
            self._save_translation(
                translation,
                save_path=save_path,
                save_format=save_format,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        return translation

    def _recalculate_syllables(
        self, translation: LyricTranslation, target_lang: str
    ) -> LyricTranslation:
        """
        重新計算翻譯結果的實際音節數

        Args:
            translation: LLM 返回的翻譯結果
            target_lang: 目標語言

        Returns:
            更新音節數後的翻譯結果
        """
        # 使用 FeatureExtractor 重新計算每行的音節數
        actual_syllable_counts = [
            self.feature_extractor._count_syllables(line, target_lang)
            for line in translation.translated_lines
        ]

        # 更新翻譯結果中的音節數
        translation.syllable_counts = actual_syllable_counts

        return translation

    def _build_prompt(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        constraints: MusicConstraints,
        feedback: Optional[str] = None,
    ) -> str:
        """構建翻譯 prompt"""
        prompt_parts = []

        if feedback:
            # 重試時包含反饋
            prompt_parts.append(f"【反饋】\n{feedback}\n")

        prompt_parts.extend(
            [
                f"【原始歌詞】({source_lang})",
                source_lyrics,
                "",
                f"【目標語言】{target_lang}",
                "",
                "【音樂約束】",
                f"- 音節數: {constraints.syllable_counts}",
            ]
        )

        if constraints.rhyme_scheme:
            prompt_parts.append(f"- 押韻方案: {constraints.rhyme_scheme}")

        if constraints.pause_positions:
            prompt_parts.append(f"- 停頓位置: {constraints.pause_positions}")

        prompt_parts.extend(["", "請翻譯並確保滿足所有約束。"])

        return "\n".join(prompt_parts)

    def _save_translation(
        self,
        translation: LyricTranslation,
        save_path: Optional[str] = None,
        save_format: str = "json",
        source_lang: str = "Unknown",
        target_lang: str = "Unknown",
    ) -> None:
        """
        保存翻譯結果

        Args:
            translation: 翻譯結果
            save_path: 保存路徑（若未提供則自動生成）
            save_format: 保存格式
            source_lang: 源語言
            target_lang: 目標語言
        """
        if save_path is None:
            # 自動生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_{source_lang}_to_{target_lang}_{timestamp}.{save_format}"
            save_path = os.path.join(self.save_dir, filename)

        # 保存
        translation.save(save_path, format=save_format)
        print(f"\n✓ 翻譯結果已保存至: {save_path}")
