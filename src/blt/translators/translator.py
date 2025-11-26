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

        # 工具箱 - ConstraintValidator 同時作為驗證器和工具箱
        self.feature_extractor = FeatureExtractor()
        self.validator = ConstraintValidator()

        # 初始化 Agent - pydantic-ai will infer Google provider from model name
        self.agent = Agent(
            model=model,
            output_type=LyricTranslation,
            system_prompt=self._get_system_prompt(),
        )

        # 從 ConstraintValidator 註冊工具供 LLM 調用
        self._register_tools_from_validator()

    def _register_tools_from_validator(self):
        """從 ConstraintValidator 註冊工具供 LLM 調用"""

        # 包裝 validator 的方法為簡潔的工具函數
        def verify_all_constraints(
            lines: list[str],
            language: str,
            target_syllables: list[int],
            rhyme_scheme: str = "",
        ) -> dict:
            """Verify all constraints at once (most efficient). Returns: {"syllables": [int], "syllables_match": bool, "rhyme_endings": [str], "rhymes_valid": bool}"""
            return self.validator.verify_all_constraints(
                lines, language, target_syllables, rhyme_scheme
            )

        def count_syllables(text: str, language: str) -> int:
            """Count syllables in single text. Use verify_all_constraints for multiple lines."""
            return self.validator.count_syllables(text, language)

        def check_rhyme(text1: str, text2: str, language: str) -> dict:
            """Check if two texts rhyme. Returns: {"rhymes": bool, "rhyme1": str, "rhyme2": str}"""
            return self.validator.check_rhyme(text1, text2, language)

        # 註冊工具到 Agent
        self.agent.tool_plain(verify_all_constraints)
        self.agent.tool_plain(count_syllables)
        self.agent.tool_plain(check_rhyme)

    def _get_system_prompt(self) -> str:
        """獲取系統 prompt"""
        return """You are a professional lyrics translation expert specialized in singable translations.

CONSTRAINT PRIORITIES (strictly enforced in this order):
1. SYLLABLE COUNT (CRITICAL) - Must match exactly
2. Rhyme scheme (IMPORTANT) - Match when possible, syllable count takes precedence
3. Pause positions (OPTIONAL) - Guidance only

EFFICIENT VERIFICATION:
- Use verify_all_constraints(lines, language, target_syllables, rhyme_scheme) to check all lines at once
- Only use count_syllables for individual line adjustments
- Tool returns: syllables, syllables_match, rhyme_endings, rhymes_valid

WORKFLOW:
1. Draft all translations (prioritize syllable count over grammar perfection)
2. Call verify_all_constraints to check entire translation
3. If syllables_match=False, identify mismatches and adjust those specific lines
4. Re-verify until syllables_match=True, then output

Limit to 3 verification rounds. If still mismatched, output best attempt with reasoning."""

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
            prompt_parts.append(f"PREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n")

        prompt_parts.extend(
            [
                f"TRANSLATE FROM {source_lang} TO {target_lang}",
                "",
                "SOURCE LYRICS:",
                source_lyrics,
                "",
                "CONSTRAINTS:",
                f"• Syllable counts per line: {constraints.syllable_counts}",
            ]
        )

        if constraints.rhyme_scheme:
            prompt_parts.append(f"• Rhyme scheme: {constraints.rhyme_scheme}")

        if constraints.pause_positions:
            prompt_parts.append(f"• Pause positions: {constraints.pause_positions}")

        prompt_parts.extend(
            [
                "",
                "Translate ensuring all constraints are met. Verify each line's syllable count using count_syllables tool.",
            ]
        )

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
            filename = (
                f"translation_{source_lang}_to_{target_lang}_{timestamp}.{save_format}"
            )
            save_path = os.path.join(self.save_dir, filename)

        # 保存
        translation.save(save_path, format=save_format)
        print(f"\n✓ 翻譯結果已保存至: {save_path}")
