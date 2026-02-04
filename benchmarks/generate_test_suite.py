#!/usr/bin/env python3
"""
Generate a fixed test suite from HuggingFace parallel data.

Creates a reproducible JSON test suite for ablation experiments.

Usage:
    python -m benchmarks.generate_test_suite --samples 100
    python -m benchmarks.generate_test_suite --samples 100 --output benchmarks/results/my_suite.json
"""

import json
import argparse
from pathlib import Path


def generate_test_suite(
    data_path: str | Path,
    source_lang: str = "en-us",
    target_lang: str = "cmn",
    split: str = "test",
    num_samples: int = 100,
    max_lines: int = 5,
) -> list[dict]:
    """Load and chunk HF parallel data into test cases."""
    data_path = Path(data_path)
    source_file = data_path / "data_parallel" / f"{split}.source"
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    test_cases = []
    for i in range(0, len(all_lines), max_lines):
        chunk = all_lines[i : i + max_lines]
        if len(chunk) < 2:
            continue
        test_cases.append(
            {
                "id": f"{source_lang}\u2192{target_lang}_hf_{i // max_lines + 1:03d}",
                "source_lines": chunk,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "metadata": {
                    "song_title": f"HF chunk {i // max_lines + 1}",
                    "artist_name": "lyric-trans-en2zh-data",
                    "line_count": len(chunk),
                    "dataset": "LongshenOu/lyric-trans-en2zh-data",
                    "split": split,
                    "line_offset": i,
                },
            }
        )

    test_cases = test_cases[:num_samples]
    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed test suite from HF data"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of test cases (default: 100)"
    )
    parser.add_argument(
        "--max-lines", type=int, default=5, help="Max lines per case (default: 5)"
    )
    parser.add_argument(
        "--hf-data",
        type=str,
        default="benchmarks/data/hf_en2zh/datasets",
        help="Path to HF datasets/ directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: benchmarks/results/en-us→cmn_ablation_n{N}_test_suite.json)",
    )
    args = parser.parse_args()

    test_cases = generate_test_suite(
        data_path=args.hf_data,
        num_samples=args.samples,
        max_lines=args.max_lines,
    )

    output_path = (
        args.output
        or f"benchmarks/results/en-us\u2192cmn_ablation_n{args.samples}_test_suite.json"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(test_cases)} test cases -> {output_path}")


if __name__ == "__main__":
    main()
