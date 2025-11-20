#!/usr/bin/env python3
"""
Grid search script to find optimal top-k and threshold parameters for ICL.
Tests different combinations and evaluates performance.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_FILE = PROJECT_ROOT / "prompts.txt"


def run_evaluation(top_k: int, threshold: float, duration: int, output_base: Path):
    """Run evaluation with specific parameters."""
    from poc import (
        get_clap_score,
        get_top_k_examples,
        run_generate_music,
    )

    # Load prompts
    with open(PROMPTS_FILE) as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Create output directory
    output_dir = output_base / f"k{top_k}_t{threshold:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSV
    csv_path = output_dir / "scores.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "prompt_id",
            "prompt",
            "baseline_score",
            "icl_score",
            "num_examples",
            "baseline_audio",
            "icl_audio",
        ]
    )

    print(f"\nRunning evaluation: top-k={top_k}, threshold={threshold}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        prompt_id = f"{idx + 1:03d}"
        print(f"[{prompt_id}/{len(prompts):03d}] {prompt[:50]}...")

        baseline_output_path = output_dir / f"{prompt_id}_baseline.wav"
        icl_output_path = output_dir / f"{prompt_id}_icl.wav"

        # Generate baseline
        run_generate_music(
            prompt=prompt,
            duration=duration,
            output_path=str(baseline_output_path),
        )
        baseline_score = get_clap_score(prompt, str(baseline_output_path))

        # Generate ICL
        examples = get_top_k_examples(prompt, k=top_k, threshold=threshold)
        num_examples = len(examples)

        if examples:
            run_generate_music(
                prompt=prompt,
                duration=duration,
                output_path=str(icl_output_path),
                examples=examples,
            )
            icl_score = get_clap_score(prompt, str(icl_output_path))
        else:
            icl_score = None

        # Write to CSV
        csv_writer.writerow(
            [
                prompt_id,
                prompt,
                f"{baseline_score:.4f}",
                f"{icl_score:.4f}" if icl_score is not None else "",
                num_examples,
                baseline_output_path.name,
                icl_output_path.name if icl_score is not None else "",
            ]
        )
        csv_file.flush()

        icl_str = f"{icl_score:.4f}" if icl_score is not None else "N/A"
        print(f"  Base={baseline_score:.4f} | ICL={icl_str} | Examples={num_examples}")

    csv_file.close()
    return csv_path


def analyze_config(csv_path: Path):
    """Analyze results for a specific configuration."""
    df = pd.read_csv(csv_path)
    df = df[df["icl_score"].notna()]  # Filter out cases with no ICL

    if len(df) == 0:
        return {
            "avg_improvement": 0,
            "win_rate": 0,
            "num_cases": 0,
            "avg_baseline": 0,
            "avg_icl": 0,
        }

    df["score_diff"] = df["icl_score"] - df["baseline_score"]
    df["improvement"] = df["score_diff"] > 0

    return {
        "avg_improvement": df["score_diff"].mean(),
        "win_rate": df["improvement"].sum() / len(df),
        "num_cases": len(df),
        "avg_baseline": df["baseline_score"].mean(),
        "avg_icl": df["icl_score"].mean(),
        "max_improvement": df["score_diff"].max(),
        "min_improvement": df["score_diff"].min(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal top-k and threshold"
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1,3,5",
        help="Comma-separated top-k values to test (default: 1,3,5)",
    )
    parser.add_argument(
        "--threshold-values",
        type=str,
        default="0.6,0.7,0.8",
        help="Comma-separated threshold values to test (default: 0.6,0.7,0.8)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration of generated music in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_search_results",
        help="Base directory for results (default: grid_search_results)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results, don't run new evaluations",
    )

    args = parser.parse_args()

    # Parse parameter values
    top_k_values = [int(k.strip()) for k in args.top_k_values.split(",")]
    threshold_values = [float(t.strip()) for t in args.threshold_values.split(",")]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GRID SEARCH FOR OPTIMAL ICL PARAMETERS")
    print("=" * 80)
    print(f"Top-k values: {top_k_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Total configurations: {len(top_k_values) * len(threshold_values)}")
    print(f"Duration: {args.duration}s")
    print("=" * 80)

    # Run grid search
    results = []

    if not args.analyze_only:
        for top_k in top_k_values:
            for threshold in threshold_values:
                csv_path = run_evaluation(top_k, threshold, args.duration, output_base)
                metrics = analyze_config(csv_path)
                results.append(
                    {
                        "top_k": top_k,
                        "threshold": threshold,
                        **metrics,
                    }
                )

        # Save summary
        summary_path = output_base / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

    # Display results
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    if args.analyze_only:
        # Load from existing summary
        summary_path = Path(args.output_dir) / "summary.json"
        if not summary_path.exists():
            print(f"Error: Summary file not found at {summary_path}")
            return
        with open(summary_path) as f:
            results = json.load(f)

    # Sort by average improvement
    results.sort(key=lambda x: x["avg_improvement"], reverse=True)

    print("\nRanked by Average Improvement:")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'Top-K':<8} {'Thresh':<8} {'Avg Δ':<10} {'Win Rate':<12} {'Cases':<8} {'Max Δ':<10} {'Min Δ':<10}"
    )
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(
            f"{i:<6} {result['top_k']:<8} {result['threshold']:<8.2f} "
            f"{result['avg_improvement']:+.4f}    {result['win_rate'] * 100:>5.1f}%       "
            f"{result['num_cases']:<8} {result['max_improvement']:+.4f}     {result['min_improvement']:+.4f}"
        )

    print("-" * 80)

    # Best configuration
    best = results[0]
    print("\nBEST CONFIGURATION:")
    print(f"  Top-K: {best['top_k']}")
    print(f"  Threshold: {best['threshold']:.2f}")
    print(f"  Average improvement: {best['avg_improvement']:+.4f}")
    print(f"  Win rate: {best['win_rate'] * 100:.1f}%")
    print(f"  Cases evaluated: {best['num_cases']}")
    print(f"  Max improvement: {best['max_improvement']:+.4f}")
    print(f"  Min degradation: {best['min_improvement']:+.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
