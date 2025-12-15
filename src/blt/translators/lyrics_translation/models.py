"""Models for lyrics translation"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class MusicConstraints(BaseModel):
    """Music constraints for lyrics translation"""

    syllable_counts: list[int] = Field(description="Target syllable count per line")
    rhyme_scheme: Optional[str] = Field(
        default=None, description="Rhyme scheme (e.g., AABB, ABAB, AAAA)"
    )
    syllable_patterns: Optional[list[list[int]]] = Field(
        default=None,
        description="Target syllable patterns per line (e.g., [[1,1,1,3], [1,3,2,4]])",
    )


class LyricTranslation(BaseModel):
    """Standard lyrics translation output"""

    translated_lines: list[str] = Field(description="Translated lyrics line by line")
    syllable_counts: list[int] = Field(
        description="Syllable count per line (LLM outputs, we recalculate)"
    )
    rhyme_endings: list[str] = Field(
        description="Rhyme ending per line (LLM outputs, we recalculate)"
    )
    syllable_patterns: Optional[list[list[int]]] = Field(
        default=None,
        description="Syllable patterns per line (LLM outputs, we recalculate)",
    )
    reasoning: str = Field(description="Translation reasoning and considerations")
    tool_call_stats: Optional[dict[str, int]] = Field(
        default=None, description="Tool call statistics: {tool_name: call_count}"
    )

    def save(self, output_path: str | Path, format: str = "json") -> None:
        """
        Save translation result to file

        Args:
            output_path: Output file path
            format: Output format ("json", "txt", "md")
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
        """Save as JSON format"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "translation": self.model_dump(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_txt(self, output_path: Path) -> None:
        """Save as plain text format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Translation Result\n")
            f.write("=" * 60 + "\n\n")

            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write(f"\nSyllable counts: {self.syllable_counts}\n")
            f.write(f"Rhyme endings: {self.rhyme_endings}\n\n")

            if self.tool_call_stats:
                f.write("Tool call statistics:\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"  - {tool_name}: {count}\n")
                f.write("\n")

            f.write(f"Translation reasoning:\n{self.reasoning}\n")

    def _save_markdown(self, output_path: Path) -> None:
        """Save as Markdown format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Translation Result\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## Translation\n\n")
            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write("\n## Music Features\n\n")
            f.write(f"- **Syllable counts**: {self.syllable_counts}\n")
            f.write(f"- **Rhyme endings**: {self.rhyme_endings}\n")

            if self.tool_call_stats:
                f.write("\n## Tool Call Statistics\n\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"- **{tool_name}**: {count}\n")

            f.write("\n## Translation Reasoning\n\n")
            f.write(f"{self.reasoning}\n")


class ValidationResult(BaseModel):
    """Validation result"""

    passed: bool = Field(description="Whether all constraints are satisfied")
    errors: list[dict] = Field(default_factory=list, description="Error list")
    score: float = Field(default=0.0, description="Overall quality score (0-1)")
