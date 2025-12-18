#!/usr/bin/env python3
"""Generate performance comparison visualizations for agent vs base methods."""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Define result directory and asset output directory
RESULTS_DIR = Path(__file__).parent / "results"
ASSETS_DIR = Path(__file__).parent.parent / "assets" / "visualization"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Language pairs with both agent and base combinations
COMBINATIONS = [
    "en-us→cmn",
    "cmn→en-us",
    "cmn→ja",
    "en-us→ja",
]


def load_json_file(file_path: Path) -> dict:
    """Load a JSON file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(results: List[dict]) -> Dict[str, float]:
    """Extract average metrics from results list."""
    metrics = {"ser": [], "scre": [], "time": []}

    for result in results:
        if "metrics" in result:
            metrics["ser"].append(result["metrics"].get("ser", 0))
            metrics["scre"].append(result["metrics"].get("scre", 0))
        if "time_seconds" in result:
            metrics["time"].append(result["time_seconds"])

    return {
        "ser": statistics.mean(metrics["ser"]) if metrics["ser"] else 0,
        "scre": statistics.mean(metrics["scre"]) if metrics["scre"] else 0,
        "time": statistics.mean(metrics["time"]) if metrics["time"] else 0,
    }


def load_combination_data() -> Tuple[Dict, Dict]:
    """Load all agent and base data for combinations with both methods."""
    agent_data = {}
    base_data = {}

    for combination in COMBINATIONS:
        agent_file = RESULTS_DIR / f"{combination}_agent.json"
        base_file = RESULTS_DIR / f"{combination}_base.json"

        if agent_file.exists() and base_file.exists():
            agent_json = load_json_file(agent_file)
            base_json = load_json_file(base_file)

            agent_metrics = extract_metrics(agent_json.get("results", []))
            base_metrics = extract_metrics(base_json.get("results", []))

            agent_data[combination] = agent_metrics
            base_data[combination] = base_metrics
            print(f"Loaded {combination}: agent and base")
        else:
            if not agent_file.exists():
                print(f"Warning: {agent_file} not found")
            if not base_file.exists():
                print(f"Warning: {base_file} not found")

    return agent_data, base_data


def create_total_performance_chart(agent_data: Dict, base_data: Dict):
    """Create a chart showing total performance metrics (agent vs base)."""
    # Calculate overall averages
    metrics_names = ["ser", "scre", "time"]

    agent_averages = {}
    base_averages = {}

    for metric in metrics_names:
        agent_values = [data[metric] for data in agent_data.values()]
        base_values = [data[metric] for data in base_data.values()]

        agent_averages[metric] = statistics.mean(agent_values) if agent_values else 0
        base_averages[metric] = statistics.mean(base_values) if base_values else 0

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Total Performance: Agent vs Base Methods", fontsize=16, fontweight="bold"
    )

    metric_labels = {
        "ser": "Syllable Error Rate (lower is better)",
        "scre": "Speech Continuity Rate Error (lower is better)",
        "time": "Average Time (seconds) (lower is better)",
    }

    for ax, metric in zip(axes, metrics_names):
        methods = ["Agent", "Base"]
        values = [agent_averages[metric], base_averages[metric]]

        bars = ax.bar(methods, values, color=["#2E86AB", "#A23B72"])
        ax.set_ylabel(metric.upper(), fontweight="bold")
        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = ASSETS_DIR / "total_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def create_by_combination_chart(agent_data: Dict, base_data: Dict):
    """Create a chart showing performance by language combination."""
    # Prepare data for comparison
    combinations_list = list(agent_data.keys())
    metrics_names = ["ser", "scre"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Performance by Language Combination: Agent vs Base",
        fontsize=16,
        fontweight="bold",
    )

    metric_labels = {
        "ser": "Syllable Error Rate (lower is better)",
        "scre": "Speech Continuity Rate Error (lower is better)",
    }

    for ax, metric in zip(axes, metrics_names):
        agent_values = [agent_data[combo][metric] for combo in combinations_list]
        base_values = [base_data[combo][metric] for combo in combinations_list]

        x = range(len(combinations_list))
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            agent_values,
            width,
            label="Agent",
            color="#2E86AB",
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            base_values,
            width,
            label="Base",
            color="#A23B72",
        )

        ax.set_ylabel(metric.upper(), fontweight="bold")
        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(combinations_list, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    output_path = ASSETS_DIR / "by_combination_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("Loading benchmark data...")
    agent_data, base_data = load_combination_data()

    if not agent_data or not base_data:
        print("Error: No data loaded. Check file paths and combinations.")
        return

    print(f"\nGenerating visualizations for {len(agent_data)} combinations...")

    # Generate visualizations
    create_total_performance_chart(agent_data, base_data)
    create_by_combination_chart(agent_data, base_data)

    print("\nVisualization generation complete!")
    print(f"Images saved to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
