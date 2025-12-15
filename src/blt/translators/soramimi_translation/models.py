"""Models for soramimi translation"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field


class SoramimiTranslationState(TypedDict):
    """State for mapping-based soramimi translation graph"""

    # Source information
    source_lines: list[str]
    source_lang: str
    target_lang: str

    # Phoneme mapping
    source_phonemes: list[str]  # Unique phonemes from source
    phoneme_mapping: dict[str, str]  # phoneme -> target character/syllable
    mapping_scores: dict[str, float]  # phoneme -> similarity score

    # Current translation
    soramimi_lines: Optional[list[str]]
    source_ipa: Optional[list[str]]
    target_ipa: Optional[list[str]]
    similarity_scores: Optional[list[float]]
    overall_similarity: Optional[float]

    # Best results
    best_mapping: Optional[dict[str, str]]
    best_lines: Optional[list[str]]
    best_scores: Optional[list[float]]
    best_ipas: Optional[list[tuple[str, str]]]

    # Control
    attempt: int
    max_attempts: int
    threshold: float
    messages: Annotated[list, add]


class SoramimiTranslation(BaseModel):
    """Soramimi (phonetic) translation output"""

    soramimi_lines: list[str] = Field(
        description="Soramimi lyrics in Chinese characters"
    )
    reasoning: str = Field(
        default="", description="Translation reasoning and considerations"
    )

    # Optional fields - filled by tools/validator
    source_ipa: Optional[list[str]] = Field(
        default=None, description="IPA transcription of source lyrics"
    )
    target_ipa: Optional[list[str]] = Field(
        default=None, description="IPA transcription of translated lyrics"
    )
    similarity_scores: Optional[list[float]] = Field(
        default=None, description="Phonetic similarity score per line (0-1)"
    )
    overall_similarity: Optional[float] = Field(
        default=None, description="Overall phonetic similarity (0-1)"
    )
    tool_call_stats: Optional[dict[str, int]] = Field(
        default=None, description="Tool call statistics: {tool_name: call_count}"
    )

    def save(self, output_path: str | Path, format: str = "json") -> None:
        """
        Save soramimi translation result to file

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
            "soramimi_translation": self.model_dump(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_txt(self, output_path: Path) -> None:
        """Save as plain text format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Soramimi Translation Result\n")
            f.write("=" * 60 + "\n\n")

            for i, (line, src_ipa, tgt_ipa, score) in enumerate(
                zip(
                    self.soramimi_lines,
                    self.source_ipa or [],
                    self.target_ipa or [],
                    self.similarity_scores or [],
                ),
                1,
            ):
                f.write(f"{i}. {line}\n")
                f.write(f"   Source IPA: {src_ipa}\n")
                f.write(f"   Target IPA: {tgt_ipa}\n")
                f.write(f"   Similarity: {score:.1%}\n\n")

            f.write(f"Overall Similarity: {self.overall_similarity:.1%}\n\n")

            if self.tool_call_stats:
                f.write("Tool call statistics:\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"  - {tool_name}: {count}\n")
                f.write("\n")

            f.write(f"Translation reasoning:\n{self.reasoning}\n")

    def _save_markdown(self, output_path: Path) -> None:
        """Save as Markdown format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Soramimi Translation Result\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## Translation\n\n")
            f.write("| # | Translation | Source IPA | Target IPA | Similarity |\n")
            f.write("|---|-------------|------------|------------|------------|\n")
            for i, (line, src_ipa, tgt_ipa, score) in enumerate(
                zip(
                    self.soramimi_lines,
                    self.source_ipa or [],
                    self.target_ipa or [],
                    self.similarity_scores or [],
                ),
                1,
            ):
                f.write(f"| {i} | {line} | {src_ipa} | {tgt_ipa} | {score:.1%} |\n")

            f.write(f"\n**Overall Similarity: {self.overall_similarity:.1%}**\n")

            if self.tool_call_stats:
                f.write("\n## Tool Call Statistics\n\n")
                for tool_name, count in self.tool_call_stats.items():
                    f.write(f"- **{tool_name}**: {count}\n")

            f.write("\n## Translation Reasoning\n\n")
            f.write(f"{self.reasoning}\n")
