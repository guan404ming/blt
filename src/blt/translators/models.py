"""
Pydantic Models for Lyrics Translation
定義結構化輸出的資料模型
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class MusicConstraints(BaseModel):
    """音樂約束條件"""

    syllable_counts: list[int] = Field(description="每行的目標音節數")
    rhyme_scheme: Optional[str] = Field(
        default=None, description="押韻方案，例如: AABB, ABAB, AAAA"
    )
    pause_positions: Optional[list[int]] = Field(
        default=None, description="音樂停頓位置，詞邊界應該對齊的位置"
    )


class LyricTranslation(BaseModel):
    """標準歌詞翻譯輸出"""

    translated_lines: list[str] = Field(description="Translated lyrics line by line")
    syllable_counts: list[int] = Field(
        description="Syllable count per line (LLM outputs, we recalculate)"
    )
    rhyme_endings: list[str] = Field(
        description="Rhyme ending per line (LLM outputs, we recalculate)"
    )
    reasoning: str = Field(description="Translation reasoning and considerations")
    tool_call_stats: Optional[dict[str, int]] = Field(
        default=None, description="Tool call statistics: {tool_name: call_count}"
    )

    def save(self, output_path: str | Path, format: str = "json") -> None:
        """
        保存翻譯結果到文件

        Args:
            output_path: 輸出文件路徑
            format: 輸出格式 ("json", "txt", "md")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._save_json(output_path)
        elif format == "txt":
            self._save_txt(output_path)
        elif format == "md":
            self._save_markdown(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(self, output_path: Path) -> None:
        """保存為 JSON 格式"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "translation": self.model_dump(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_txt(self, output_path: Path) -> None:
        """保存為純文本格式"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("翻譯結果\n")
            f.write("=" * 60 + "\n\n")

            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write(f"\n音節數: {self.syllable_counts}\n")
            f.write(f"韻腳: {self.rhyme_endings}\n\n")

            if self.tool_call_stats:
                f.write("工具調用統計:\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"  - {tool_name}: {count}\n")
                f.write("\n")

            f.write(f"翻譯思路:\n{self.reasoning}\n")

    def _save_markdown(self, output_path: Path) -> None:
        """保存為 Markdown 格式"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 翻譯結果\n\n")
            f.write(f"*生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## 譯文\n\n")
            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write("\n## 音樂特徵\n\n")
            f.write(f"- **音節數**: {self.syllable_counts}\n")
            f.write(f"- **韻腳**: {self.rhyme_endings}\n")

            if self.tool_call_stats:
                f.write("\n## 工具調用統計\n\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"- **{tool_name}**: {count}\n")

            f.write("\n## 翻譯思路\n\n")
            f.write(f"{self.reasoning}\n")


class ValidationResult(BaseModel):
    """驗證結果"""

    passed: bool = Field(description="是否通過所有約束驗證")
    errors: list[dict] = Field(default_factory=list, description="錯誤列表")
    score: float = Field(default=0.0, description="整體品質評分 (0-1)")
