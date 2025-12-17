"""
Experiment Utilities

Helper functions for preparing test cases from scraped data.
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from .runner import TestCase


def _detect_language(text: str) -> str:
    """
    Simple language detection based on character composition
    """
    # Count character types
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))

    total = chinese_chars + japanese_chars + english_chars
    if total == 0:
        return "unknown"

    # Determine primary language
    if chinese_chars / total > 0.5:
        return "cmn"
    elif japanese_chars / total > 0.3:
        return "ja"
    elif english_chars / total > 0.5:
        return "en-us"
    else:
        return "mixed"


def load_lyrics_from_json(
    filepath: str | Path,
    language: str,
) -> list[dict]:
    """
    Load lyrics from scraped JSON file

    Args:
        filepath: Path to lyrics JSON file
        language: Language code (cmn, en-us, ja)

    Returns:
        List of song dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add language if not present
    for song in data:
        if "language" not in song:
            song["language"] = language

    return data


def sample_test_cases(
    source_lyrics: list[dict],
    source_lang: str,
    target_lang: str,
    num_samples: int | None = 10,
    min_lines: int = 4,
    max_lines: int = 20,
) -> list[TestCase]:
    """
    Create test cases from lyrics dataset (sequential, no randomization)

    Args:
        source_lyrics: List of song dictionaries
        source_lang: Source language code
        target_lang: Target language code
        num_samples: Number of test cases to create (None = all)
        min_lines: Minimum number of lines per sample
        max_lines: Maximum number of lines per sample

    Returns:
        List of TestCase objects
    """
    # Filter songs with enough lines
    valid_songs = [
        song for song in source_lyrics if song.get("line_count", 0) >= min_lines
    ]

    # If num_samples is None, use all valid songs
    if num_samples is None:
        num_samples = len(valid_songs)
    elif len(valid_songs) < num_samples:
        print(f"Warning: Only {len(valid_songs)} valid songs, requested {num_samples}")
        num_samples = len(valid_songs)

    # Take first num_samples songs (sequential, no randomization)
    sampled_songs = valid_songs[:num_samples]

    test_cases = []
    for i, song in enumerate(sampled_songs, 1):
        # Get lyrics lines
        lyrics = song["lyrics"]
        raw_lines = [line.strip() for line in lyrics.split("\n") if line.strip()]

        # Filter lines by language
        lines = []
        for line in raw_lines:
            detected = _detect_language(line)
            # If detected as a specific language that isn't the source, skip it
            if (
                detected != "unknown"
                and detected != "mixed"
                and detected != source_lang
            ):
                continue
            lines.append(line)

        # Skip song if not enough lines after filtering
        if len(lines) < min_lines:
            continue

        # Truncate if too long
        if len(lines) > max_lines:
            # Take first max_lines
            lines = lines[:max_lines]

        test_case: TestCase = {
            "id": f"{source_lang}→{target_lang}_{i:03d}",
            "source_lines": lines,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "metadata": {
                "song_title": song.get("song_title", "Unknown"),
                "artist_name": song.get("artist_name", "Unknown"),
                "line_count": len(lines),
            },
        }

        test_cases.append(test_case)

    return test_cases


def create_test_suite(
    data_dir: str | Path = "benchmarks/data",
    language_pairs: list[tuple[str, str]] | None = None,
    samples_per_pair: int = 10,
) -> dict[str, list[TestCase]]:
    """
    Create test suites for multiple language pairs

    Args:
        data_dir: Directory with scraped lyrics data
        language_pairs: List of (source, target) language pairs
        samples_per_pair: Number of samples per language pair

    Returns:
        Dict mapping language pair to test cases
    """
    data_dir = Path(data_dir)

    # Default language pairs
    if language_pairs is None:
        language_pairs = [
            ("cmn", "en-us"),
            ("cmn", "ja"),
            ("en-us", "cmn"),
            ("en-us", "ja"),
            ("ja", "cmn"),
            ("ja", "en-us"),
        ]

    # Load datasets
    datasets = {}
    for lang_file in ["cmn_lyrics.json", "en_lyrics.json", "ja_lyrics.json"]:
        filepath = data_dir / lang_file
        if filepath.exists():
            lang_code = lang_file.split("_")[0]
            if lang_code == "en":
                lang_code = "en-us"
            datasets[lang_code] = load_lyrics_from_json(filepath, lang_code)
        else:
            print(f"Warning: {filepath} not found")

    # Create test cases for each pair
    test_suites = {}
    for source_lang, target_lang in language_pairs:
        if source_lang not in datasets:
            print(
                f"Warning: No data for {source_lang}, skipping {source_lang}→{target_lang}"
            )
            continue

        pair_key = f"{source_lang}→{target_lang}"
        test_cases = sample_test_cases(
            source_lyrics=datasets[source_lang],
            source_lang=source_lang,
            target_lang=target_lang,
            num_samples=samples_per_pair,
        )

        test_suites[pair_key] = test_cases
        print(f"Created {len(test_cases)} test cases for {pair_key}")

    return test_suites


def save_test_suite(
    test_cases: list[TestCase],
    output_path: str | Path,
) -> None:
    """
    Save test suite to JSON

    Args:
        test_cases: List of test cases
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"💾 Test suite saved to: {output_path}")


def load_test_suite(filepath: str | Path) -> list[TestCase]:
    """
    Load test suite from JSON

    Args:
        filepath: Path to test suite JSON

    Returns:
        List of TestCase objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
