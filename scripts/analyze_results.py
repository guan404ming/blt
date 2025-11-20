#!/usr/bin/env python3
"""
Script to analyze CLAP score results from evaluation CSV files.
"""

import argparse
from pathlib import Path

import pandas as pd

# Optional plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def analyze_scores(csv_path: Path, output_dir: Path = None):
    """Analyze CLAP scores from CSV file.

    Args:
        csv_path: Path to the scores CSV file
        output_dir: Optional directory to save plots
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Calculate score differences
    df["score_diff"] = df["icl_score"] - df["baseline_score"]
    df["improvement"] = df["score_diff"] > 0

    # Print analysis
    print("=" * 80)
    print("CLAP SCORE ANALYSIS: Baseline vs In-Context Learning (ICL)")
    print("=" * 80)
    print()

    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total prompts evaluated: {len(df)}")
    print(
        f"Average baseline score:  {df['baseline_score'].mean():.4f} ± {df['baseline_score'].std():.4f}"
    )
    print(
        f"Average ICL score:       {df['icl_score'].mean():.4f} ± {df['icl_score'].std():.4f}"
    )
    print(f"Average difference:      {df['score_diff'].mean():.4f}")
    print()

    # Win/Loss analysis
    improved = df["improvement"].sum()
    degraded = (~df["improvement"]).sum()
    print("WIN/LOSS ANALYSIS")
    print("-" * 80)
    print(f"ICL improved:  {improved}/{len(df)} ({improved / len(df) * 100:.1f}%)")
    print(f"ICL degraded:  {degraded}/{len(df)} ({degraded / len(df) * 100:.1f}%)")
    print()

    # Best improvements
    print("TOP 5 IMPROVEMENTS (ICL > Baseline)")
    print("-" * 80)
    top_improvements = df.nlargest(5, "score_diff")[
        ["prompt_id", "prompt", "baseline_score", "icl_score", "score_diff"]
    ]
    for _, row in top_improvements.iterrows():
        print(
            f"#{row['prompt_id']:03d} | Δ={row['score_diff']:+.4f} | "
            f"Base={row['baseline_score']:.4f} → ICL={row['icl_score']:.4f}"
        )
        print(f"      {row['prompt'][:70]}...")
        print()

    # Worst degradations
    print("TOP 5 DEGRADATIONS (ICL < Baseline)")
    print("-" * 80)
    top_degradations = df.nsmallest(5, "score_diff")[
        ["prompt_id", "prompt", "baseline_score", "icl_score", "score_diff"]
    ]
    for _, row in top_degradations.iterrows():
        print(
            f"#{row['prompt_id']:03d} | Δ={row['score_diff']:+.4f} | "
            f"Base={row['baseline_score']:.4f} → ICL={row['icl_score']:.4f}"
        )
        print(f"      {row['prompt'][:70]}...")
        print()

    # Score ranges
    print("SCORE DISTRIBUTION")
    print("-" * 80)
    print("Baseline scores:")
    print(f"  Min:  {df['baseline_score'].min():.4f}")
    print(f"  25%:  {df['baseline_score'].quantile(0.25):.4f}")
    print(f"  50%:  {df['baseline_score'].median():.4f}")
    print(f"  75%:  {df['baseline_score'].quantile(0.75):.4f}")
    print(f"  Max:  {df['baseline_score'].max():.4f}")
    print()
    print("ICL scores:")
    print(f"  Min:  {df['icl_score'].min():.4f}")
    print(f"  25%:  {df['icl_score'].quantile(0.25):.4f}")
    print(f"  50%:  {df['icl_score'].median():.4f}")
    print(f"  75%:  {df['icl_score'].quantile(0.75):.4f}")
    print(f"  Max:  {df['icl_score'].max():.4f}")
    print()

    print("=" * 80)

    # Generate plots if output directory is specified
    if output_dir:
        if not PLOTTING_AVAILABLE:
            print("\nWarning: matplotlib and seaborn not installed. Skipping plots.")
            print("Install with: uv pip install matplotlib seaborn")
            return df
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

        # Plot 1: Score comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        x = range(len(df))
        ax.plot(x, df["baseline_score"], "o-", label="Baseline", linewidth=2)
        ax.plot(x, df["icl_score"], "s-", label="ICL", linewidth=2)
        ax.set_xlabel("Prompt ID")
        ax.set_ylabel("CLAP Score")
        ax.set_title("CLAP Score Comparison: Baseline vs ICL")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "score_comparison.png", dpi=150)
        plt.close()

        # Plot 2: Score difference distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ["green" if x > 0 else "red" for x in df["score_diff"]]
        ax.bar(range(len(df)), df["score_diff"], color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Prompt ID")
        ax.set_ylabel("Score Difference (ICL - Baseline)")
        ax.set_title("ICL Impact: Score Differences")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "score_differences.png", dpi=150)
        plt.close()

        # Plot 3: Distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(
            df["baseline_score"], bins=15, alpha=0.7, label="Baseline", color="blue"
        )
        axes[0].hist(df["icl_score"], bins=15, alpha=0.7, label="ICL", color="orange")
        axes[0].set_xlabel("CLAP Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Score Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(
            [df["baseline_score"], df["icl_score"]],
            labels=["Baseline", "ICL"],
            patch_artist=True,
        )
        axes[1].set_ylabel("CLAP Score")
        axes[1].set_title("Score Distribution (Box Plot)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "score_distributions.png", dpi=150)
        plt.close()

        # Plot 4: Scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(df["baseline_score"], df["icl_score"], alpha=0.6, s=100)

        # Add diagonal line (y=x)
        min_val = min(df["baseline_score"].min(), df["icl_score"].min())
        max_val = max(df["baseline_score"].max(), df["icl_score"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")

        ax.set_xlabel("Baseline Score")
        ax.set_ylabel("ICL Score")
        ax.set_title("Baseline vs ICL Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_plot.png", dpi=150)
        plt.close()

        print(f"\nPlots saved to: {output_dir}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CLAP score results from evaluation CSV"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the scores CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis plots (optional)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    analyze_scores(csv_path, output_dir)


if __name__ == "__main__":
    main()
