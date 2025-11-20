#!/usr/bin/env python3
"""
Script to find the optimal baseline score threshold for using ICL.
Analyzes when ICL helps vs hurts based on baseline scores.
"""

import argparse
from pathlib import Path

import pandas as pd


def find_optimal_threshold(csv_path: Path):
    """Find optimal threshold for when to use ICL.

    Args:
        csv_path: Path to the scores CSV file
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    df["score_diff"] = df["icl_score"] - df["baseline_score"]
    df["improvement"] = df["score_diff"] > 0

    print("=" * 80)
    print("FINDING OPTIMAL THRESHOLD FOR ICL")
    print("=" * 80)
    print()

    # Analyze by baseline score ranges
    print("PERFORMANCE BY BASELINE SCORE RANGE")
    print("-" * 80)

    ranges = [
        (0.0, 0.3, "Very Low (0.0-0.3)"),
        (0.3, 0.5, "Low (0.3-0.5)"),
        (0.5, 0.6, "Medium (0.5-0.6)"),
        (0.6, 0.7, "High (0.6-0.7)"),
        (0.7, 1.0, "Very High (0.7-1.0)"),
    ]

    for min_score, max_score, label in ranges:
        mask = (df["baseline_score"] >= min_score) & (df["baseline_score"] < max_score)
        subset = df[mask]

        if len(subset) == 0:
            continue

        avg_diff = subset["score_diff"].mean()
        improved = subset["improvement"].sum()
        total = len(subset)
        improvement_rate = improved / total if total > 0 else 0

        print(f"\n{label}:")
        print(f"  Count:        {total}")
        print(f"  Avg baseline: {subset['baseline_score'].mean():.4f}")
        print(f"  Avg ICL:      {subset['icl_score'].mean():.4f}")
        print(f"  Avg diff:     {avg_diff:+.4f}")
        print(f"  Win rate:     {improved}/{total} ({improvement_rate*100:.1f}%)")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    # Find where ICL helps most
    improved_cases = df[df["improvement"]]
    degraded_cases = df[~df["improvement"]]

    print("ICL IMPROVES when baseline score is:")
    if len(improved_cases) > 0:
        print(f"  Average:  {improved_cases['baseline_score'].mean():.4f}")
        print(f"  Median:   {improved_cases['baseline_score'].median():.4f}")
        print(
            f"  Range:    {improved_cases['baseline_score'].min():.4f} - {improved_cases['baseline_score'].max():.4f}"
        )
    print()

    print("ICL DEGRADES when baseline score is:")
    if len(degraded_cases) > 0:
        print(f"  Average:  {degraded_cases['baseline_score'].mean():.4f}")
        print(f"  Median:   {degraded_cases['baseline_score'].median():.4f}")
        print(
            f"  Range:    {degraded_cases['baseline_score'].min():.4f} - {degraded_cases['baseline_score'].max():.4f}"
        )
    print()

    # Find optimal threshold
    optimal_threshold = (
        improved_cases["baseline_score"].max() if len(improved_cases) > 0 else 0.5
    )

    print(f"OPTIMAL STRATEGY:")
    print(f"  Use ICL when baseline CLAP score < {optimal_threshold:.2f}")
    print(f"  Skip ICL when baseline CLAP score >= {optimal_threshold:.2f}")
    print()

    # Calculate potential improvement with this strategy
    would_use_icl = df["baseline_score"] < optimal_threshold
    correct_decisions = (would_use_icl & df["improvement"]) | (
        ~would_use_icl & ~df["improvement"]
    )

    print(f"If this strategy were applied:")
    print(f"  Would use ICL: {would_use_icl.sum()}/{len(df)} cases")
    print(
        f"  Correct decisions: {correct_decisions.sum()}/{len(df)} ({correct_decisions.sum()/len(df)*100:.1f}%)"
    )

    # Show which prompts should use ICL
    print()
    print("-" * 80)
    print("PROMPTS THAT SHOULD USE ICL (baseline < {:.2f}):".format(optimal_threshold))
    print("-" * 80)
    should_use_icl = df[df["baseline_score"] < optimal_threshold].sort_values(
        "score_diff", ascending=False
    )
    for _, row in should_use_icl.head(10).iterrows():
        status = "✓" if row["improvement"] else "✗"
        print(
            f"{status} #{row['prompt_id']:03d} | Base={row['baseline_score']:.4f} | Δ={row['score_diff']:+.4f}"
        )
        print(f"   {row['prompt'][:70]}...")

    print()
    print("-" * 80)
    print(
        "PROMPTS THAT SHOULD SKIP ICL (baseline >= {:.2f}):".format(optimal_threshold)
    )
    print("-" * 80)
    should_skip_icl = df[df["baseline_score"] >= optimal_threshold].sort_values(
        "baseline_score", ascending=False
    )
    for _, row in should_skip_icl.head(10).iterrows():
        saved = -row["score_diff"] if row["score_diff"] < 0 else 0
        status = "✓" if not row["improvement"] else "✗"
        print(
            f"{status} #{row['prompt_id']:03d} | Base={row['baseline_score']:.4f} | Saved={saved:.4f}"
        )
        print(f"   {row['prompt'][:70]}...")

    print()
    print("=" * 80)

    return optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal threshold for when to use ICL"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the scores CSV file",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    optimal_threshold = find_optimal_threshold(csv_path)
    print(f"\nOptimal threshold: {optimal_threshold:.2f}")


if __name__ == "__main__":
    main()
