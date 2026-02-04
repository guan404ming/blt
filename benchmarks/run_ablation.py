#!/usr/bin/env python3
"""
Run Phase Ablation Experiments

Compares pipeline performance across different phase configurations:
  - Phase 1 only: initial translation with tools, no refinement
  - Phase 1+2: initial translation + syllable refinement (no pattern refinement)
  - Phase 1+2+3: full pipeline (current system)

Each phase runs as a separate subprocess to avoid OOM from accumulated memory.

Usage:
    python -m benchmarks.run_ablation en-us cmn
    python -m benchmarks.run_ablation en-us cmn --samples 30
    python -m benchmarks.run_ablation en-us cmn --hf-data benchmarks/data/hf_en2zh/datasets
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


PHASE_LABELS = {
    1: "Phase 1 only (initial translation)",
    2: "Phase 1+2 (+ syllable refinement)",
    3: "Phase 1+2+3 (full pipeline)",
}


def load_hf_parallel_data(
    data_path: str | Path,
    source_lang: str,
    target_lang: str,
    split: str = "test",
    num_samples: int | None = 5,
    max_lines: int = 5,
) -> list[dict]:
    """
    Load test cases from HuggingFace lyric-trans-en2zh-data parallel format.

    The dataset has one lyric line per file line (no song boundaries).
    We chunk consecutive lines into groups of `max_lines` as test cases.
    """
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
        test_case = {
            "id": f"{source_lang}→{target_lang}_hf_{i // max_lines + 1:03d}",
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
        test_cases.append(test_case)

    if num_samples is not None:
        test_cases = test_cases[:num_samples]

    return test_cases


def run_single_phase(
    phases: int,
    test_suite_path: str,
    source_lang: str,
    target_lang: str,
    model: str,
    output_dir: str,
    max_retries: int = 10,
) -> int:
    """Run a single phase experiment as a subprocess via run_experiment.py.

    Retries on crash (e.g. OOM) up to max_retries times.
    Checkpoint auto-resume means each retry picks up where it left off.
    """
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.run_experiment",
        source_lang,
        target_lang,
        "--mode",
        "agent",
        "--phases",
        str(phases),
        "--model",
        model,
        "--test-suite",
        test_suite_path,
    ]

    env = os.environ.copy()
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

    for attempt in range(1, max_retries + 1):
        print(f"\n{'─' * 50}")
        print(f"Running: {PHASE_LABELS[phases]} (attempt {attempt}/{max_retries})")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'─' * 50}")

        result = subprocess.run(cmd, env=env)

        if result.returncode == 0:
            return 0

        print(
            f"Phase {phases} crashed (exit code {result.returncode}), attempt {attempt}/{max_retries}"
        )

        if attempt < max_retries:
            # Check if checkpoint exists — if so, retry will auto-resume
            pair_key = f"{source_lang}→{target_lang}"
            phase_suffix = f"_p{phases}" if phases != 3 else ""
            checkpoint = Path(output_dir) / f"{pair_key}_agent{phase_suffix}.json"
            if checkpoint.exists():
                print(f"Checkpoint found at {checkpoint}, retrying with auto-resume...")
            else:
                print("No checkpoint found, retrying from scratch...")

    print(f"Phase {phases} failed after {max_retries} attempts")
    return result.returncode


def collect_results(output_dir: Path, pair_key: str) -> dict:
    """Load results from saved JSON files for each phase."""
    results = {}
    for phases in [1, 2, 3]:
        # Try both hf and non-hf naming
        for suffix in [f"_p{phases}", ""]:
            filepath = output_dir / f"{pair_key}_agent{suffix}.json"
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results[phases] = data
                break
    return results


def print_summary(all_results: dict):
    """Print comparison summary table with confidence intervals when available."""
    print(f"\n{'=' * 90}")
    print("ABLATION COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Phase':<35} {'n':>4} {'SER ↓':>14} {'SCRE ↓':>14} {'ARI ↑':>14} {'Time':>8}"
    )
    print(f"{'─' * 90}")

    for phases in [1, 2, 3]:
        if phases not in all_results:
            print(f"{PHASE_LABELS[phases]:<35} {'':>4} {'(missing)':>14}")
            continue

        data = all_results[phases]
        metrics = data.get("agent_avg_metrics", {})
        if not metrics:
            print(f"{PHASE_LABELS[phases]:<35} {'':>4} {'(no data)':>14}")
            continue

        def fmt_ci(key):
            val = metrics.get(key)
            ci = metrics.get(f"{key}_ci95")
            if val is None:
                return "N/A"
            if ci is not None:
                return f"{val:.4f}±{ci:.4f}"
            return f"{val:.4f}"

        n = metrics.get("ser_n", "?")
        time_s = (
            f"{metrics.get('avg_time_seconds', 0):.1f}s"
            if isinstance(metrics.get("avg_time_seconds"), (int, float))
            else "N/A"
        )
        print(
            f"{PHASE_LABELS[phases]:<35} {n:>4} {fmt_ci('ser'):>14} {fmt_ci('scre'):>14} {fmt_ci('ari'):>14} {time_s:>8}"
        )


def save_ablation_summary(
    all_results: dict, source_lang: str, target_lang: str, model: str, output_dir: Path
) -> Path:
    """Save ablation summary as JSON."""
    summary = {
        "experiment_type": "phase_ablation",
        "timestamp": datetime.now().isoformat(),
        "language_pair": f"{source_lang}→{target_lang}",
        "model": model,
        "phases": {},
    }

    for phases, data in all_results.items():
        summary["phases"][f"p{phases}"] = {
            "label": PHASE_LABELS[phases],
            "total_tests": data.get("total_tests", 0),
            "avg_metrics": data.get("agent_avg_metrics", {}),
        }

    filepath = output_dir / f"{source_lang}→{target_lang}_ablation_summary.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nAblation summary saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run phase ablation experiments (each phase in separate process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("source_lang", help="Source language (cmn, en-us, ja)")
    parser.add_argument("target_lang", help="Target language (cmn, en-us, ja)")
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of test samples (default: 5)"
    )
    parser.add_argument(
        "--max-lines", type=int, default=5, help="Max lines per test case (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--data-dir", type=str, default="benchmarks/data", help="Lyrics data directory"
    )
    parser.add_argument(
        "--hf-data",
        type=str,
        default=None,
        help="Path to extracted HF datasets/ directory",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Max retries per phase on crash (default: 10)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        default=None,
        help="Path to pre-generated test suite JSON (skips data loading)",
    )

    args = parser.parse_args()

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_key = f"{args.source_lang}→{args.target_lang}"
    print(f"\n{'=' * 60}")
    print(f"Phase Ablation Study: {pair_key}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print(f"{'=' * 60}")

    # Step 1: Prepare test cases and save as test suite JSON
    if args.test_suite:
        # Use pre-generated test suite directly
        test_suite_path = Path(args.test_suite)
        if not test_suite_path.exists():
            print(f"Error: test suite not found: {test_suite_path}")
            return
        with open(test_suite_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(
            f"Loaded pre-generated test suite: {test_suite_path} ({len(test_cases)} cases)"
        )
    elif args.hf_data:
        test_cases = load_hf_parallel_data(
            data_path=args.hf_data,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            split="test",
            num_samples=args.samples,
            max_lines=args.max_lines,
        )
    else:
        # Use run_experiment's built-in data loading by not passing --test-suite
        # But for consistency, let's generate and save test cases
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
        from benchmarks.experiments import load_lyrics_from_json, sample_test_cases

        source_file_map = {
            "cmn": "cmn_lyrics.json",
            "en-us": "en_lyrics.json",
            "ja": "ja_lyrics.json",
        }
        source_file = Path(args.data_dir) / source_file_map.get(
            args.source_lang, f"{args.source_lang}_lyrics.json"
        )
        if not source_file.exists():
            print(f"Error: {source_file} not found")
            return
        source_lyrics = load_lyrics_from_json(source_file, args.source_lang)
        test_cases = sample_test_cases(
            source_lyrics=source_lyrics,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            num_samples=args.samples,
            max_lines=args.max_lines,
        )

    if not test_cases:
        print("No test cases found.")
        return

    print(f"Test cases prepared: {len(test_cases)}")

    # Save test suite for subprocess reuse (unless already using a pre-generated one)
    if args.test_suite:
        test_suite_path = Path(args.test_suite)
    else:
        test_suite_path = output_dir / f"{pair_key}_ablation_test_suite.json"
        with open(test_suite_path, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
        print(f"Test suite saved to: {test_suite_path}")

    # Step 2: Run each phase as a separate subprocess
    for phases in [1, 2, 3]:
        returncode = run_single_phase(
            phases=phases,
            test_suite_path=str(test_suite_path),
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            model=args.model,
            output_dir=str(output_dir),
            max_retries=args.max_retries,
        )
        if returncode != 0:
            print(f"Phase {phases} failed after all retries, continuing...")

    # Step 3: Collect results and print summary
    # Results are saved by run_experiment.py with naming: {pair_key}_agent_p{phases}.json
    all_results = {}
    for phases in [1, 2, 3]:
        for pattern in [
            f"{pair_key}_agent_p{phases}.json",
            f"{pair_key}_agent.json",
        ]:
            filepath = output_dir / pattern
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_results[phases] = data
                break

    if all_results:
        print_summary(all_results)
        save_ablation_summary(
            all_results, args.source_lang, args.target_lang, args.model, output_dir
        )
    else:
        print("No results found to summarize.")


if __name__ == "__main__":
    main()
