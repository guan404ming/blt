"""Models for Gemini lyrics translation"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field


class GeminiTranslationState(TypedDict):
    """State for Gemini lyrics translation graph"""

    # Input
    source_lyrics: str
    source_lang: str
    target_lang: str

    # Constraints
    syllable_counts: Optional[list[int]]
    rhyme_scheme: Optional[str]
    syllable_patterns: Optional[list[list[int]]]

    # Translation
    translated_lines: Optional[list[str]]
    reasoning: Optional[str]

    # Metrics
    translation_syllable_counts: Optional[list[int]]
    translation_rhyme_scheme: Optional[str]

    # Validation
    validation_passed: Optional[bool]
    validation_details: Optional[dict]

    # Control
    attempt: int
    max_attempts: int
    messages: Annotated[list, add]


class GeminiTranslation(BaseModel):
    """Gemini translation output model"""

    translated_lines: list[str] = Field(description="Translated lyrics line by line")
    syllable_counts: list[int] = Field(description="Syllable count per line")
    rhyme_scheme: str = Field(default="", description="Rhyme scheme (e.g., ABCDAECDD)")
    validation: dict = Field(default_factory=dict, description="Validation results")
    reasoning: str = Field(default="", description="Translation reasoning")

    def save(self, output_path: str | Path, format: str = "json") -> None:
        """Save translation result to file

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
            f.write("Gemini Translation Result\n")
            f.write("=" * 60 + "\n\n")

            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write(f"\nSyllable counts: {self.syllable_counts}\n")
            f.write(f"Rhyme scheme: {self.rhyme_scheme}\n\n")

            f.write("Validation Results:\n")
            for key, value in self.validation.items():
                f.write(f"  - {key}: {value}\n")

            f.write(f"\nReasoning:\n{self.reasoning}\n")

    def _save_markdown(self, output_path: Path) -> None:
        """Save as Markdown format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Gemini Translation Result\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## Translation\n\n")
            for i, line in enumerate(self.translated_lines, 1):
                f.write(f"{i}. {line}\n")

            f.write("\n## Music Features\n\n")
            f.write(f"- **Syllable counts**: {self.syllable_counts}\n")
            f.write(f"- **Rhyme scheme**: {self.rhyme_scheme}\n")

            f.write("\n## Validation\n\n")
            for key, value in self.validation.items():
                f.write(f"- **{key}**: {value}\n")

            f.write("\n## Reasoning\n\n")
            f.write(f"{self.reasoning}\n")
