"""
Lyrics Translator using PydanticAI + Gemini 2.0 Flash
核心翻譯器實作
"""

import os
from datetime import datetime
from typing import Optional
from pydantic_ai import Agent

from .models import LyricTranslation, CoTTranslation, MusicConstraints
from .feature_extractor import FeatureExtractor
from .validator import ConstraintValidator


class LyricsTranslator:
    """歌詞翻譯器"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        use_cot: bool = False,
        max_retries: int = 3,
        auto_save: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        初始化翻譯器

        Args:
            model: Gemini 模型名稱
            api_key: Google AI API Key (若未提供則從環境變數讀取)
            use_cot: 是否使用 Chain-of-Thought
            max_retries: 約束不滿足時的最大重試次數
            auto_save: 是否自動保存翻譯結果
            save_dir: 保存目錄（若未提供則使用 'outputs'）
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")

        # Set API key in environment for pydantic-ai
        os.environ["GOOGLE_API_KEY"] = self.api_key

        # 選擇輸出模型
        self.result_type = CoTTranslation if use_cot else LyricTranslation
        self.use_cot = use_cot
        self.max_retries = max_retries
        self.auto_save = auto_save
        self.save_dir = save_dir or "outputs"

        # 初始化 Agent - pydantic-ai will infer Google provider from model name
        self.agent = Agent(
            model=model,
            output_type=self.result_type,
            system_prompt=self._get_system_prompt(),
        )

        # 工具
        self.feature_extractor = FeatureExtractor()
        self.validator = ConstraintValidator()

    def _get_system_prompt(self) -> str:
        """獲取系統 prompt"""
        if self.use_cot:
            return """你是專業的歌詞翻譯專家。

請按照以下步驟進行翻譯:
1. 理解原文的核心意義和情感
2. 分析音樂約束 (音節數、押韻、停頓)
3. 構思符合約束的關鍵詞
4. 組裝完整譯文
5. 驗證是否滿足所有約束

請以結構化格式輸出每個步驟的結果。"""
        else:
            return """你是專業的歌詞翻譯專家。

請將歌詞翻譯成目標語言，並遵守以下要求:
1. 保持原意和情感
2. 符合目標語言的自然表達
3. 嚴格遵守音節數限制
4. 在指定位置押韻
5. 避免在音樂停頓處斷詞

請以結構化格式輸出翻譯結果。"""

    def translate(
        self,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
        constraints: Optional[MusicConstraints] = None,
        auto_retry: bool = True,
        save_path: Optional[str] = None,
        save_format: str = "json",
    ) -> LyricTranslation | CoTTranslation:
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
            LyricTranslation 或 CoTTranslation: 翻譯結果
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

        # 5. 驗證與重試
        if auto_retry:
            translation = self._validate_and_retry(
                translation, constraints, source_lyrics, source_lang, target_lang
            )

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
        self, translation: LyricTranslation | CoTTranslation, target_lang: str
    ) -> LyricTranslation | CoTTranslation:
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

    def _validate_and_retry(
        self,
        translation: LyricTranslation | CoTTranslation,
        constraints: MusicConstraints,
        source_lyrics: str,
        source_lang: str,
        target_lang: str,
    ) -> LyricTranslation | CoTTranslation:
        """驗證翻譯並在必要時重試"""
        # 轉換為標準格式以進行驗證
        if isinstance(translation, CoTTranslation):
            lyric_translation = LyricTranslation(
                translated_lines=translation.translated_lines,
                syllable_counts=translation.syllable_counts,
                rhyme_endings=translation.rhyme_endings,
                reasoning=translation.meaning_analysis,
                constraint_satisfaction={},
            )
        else:
            lyric_translation = translation

        # 驗證
        self.validator.target_lang = target_lang
        validation_result = self.validator.validate(lyric_translation, constraints)

        # 如果通過或達到最大重試次數，返回結果
        retry_count = 0
        while not validation_result.passed and retry_count < self.max_retries:
            retry_count += 1
            print(f"約束不滿足，正在重試 ({retry_count}/{self.max_retries})...")

            # 生成反饋
            feedback = self.validator.generate_feedback(validation_result)
            print(feedback)

            # 重新生成 prompt
            user_prompt = self._build_prompt(
                source_lyrics=source_lyrics,
                source_lang=source_lang,
                target_lang=target_lang,
                constraints=constraints,
                feedback=feedback,
            )

            # 重試
            result = self.agent.run_sync(user_prompt)
            translation = result.output

            # 重新計算音節數
            translation = self._recalculate_syllables(translation, target_lang)

            # 重新驗證
            if isinstance(translation, CoTTranslation):
                lyric_translation = LyricTranslation(
                    translated_lines=translation.translated_lines,
                    syllable_counts=translation.syllable_counts,
                    rhyme_endings=translation.rhyme_endings,
                    reasoning=translation.meaning_analysis,
                    constraint_satisfaction={},
                )
            else:
                lyric_translation = translation

            validation_result = self.validator.validate(lyric_translation, constraints)

        if validation_result.passed:
            print("✓ 所有約束都已滿足")
        else:
            print(f"⚠ 達到最大重試次數，最終得分: {validation_result.score:.2%}")

        return translation

    def _save_translation(
        self,
        translation: LyricTranslation | CoTTranslation,
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
            mode = "cot" if self.use_cot else "standard"
            filename = f"translation_{source_lang}_to_{target_lang}_{mode}_{timestamp}.{save_format}"
            save_path = os.path.join(self.save_dir, filename)

        # 保存
        translation.save(save_path, format=save_format)
        print(f"\n✓ 翻譯結果已保存至: {save_path}")
