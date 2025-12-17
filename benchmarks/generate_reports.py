#!/usr/bin/env python3
"""
Generate comparison reports for all language pairs
"""

import json
from pathlib import Path
from datetime import datetime


def calculate_metrics(results):
    """Calculate average metrics from results list"""
    if not results:
        return {"ser": 0, "scre": 0, "ari": 0, "avg_time_seconds": 0}

    metrics = {}
    ser_vals = [r.get("metrics", {}).get("ser", 0) for r in results]
    scre_vals = [r.get("metrics", {}).get("scre", 0) for r in results]
    ari_vals = [r.get("metrics", {}).get("ari", 0) for r in results]
    time_vals = [r.get("time_seconds", 0) for r in results]

    metrics["ser"] = sum(ser_vals) / len(ser_vals) if ser_vals else 0
    metrics["scre"] = sum(scre_vals) / len(scre_vals) if scre_vals else 0
    metrics["ari"] = sum(ari_vals) / len(ari_vals) if ari_vals else 0
    metrics["avg_time_seconds"] = sum(time_vals) / len(time_vals) if time_vals else 0

    return metrics


def generate_report(agent_file, base_file, report_path):
    """Generate comparison report for a language pair"""

    try:
        with open(agent_file) as f:
            agent_data = json.load(f)
        with open(base_file) as f:
            base_data = json.load(f)
    except FileNotFoundError as e:
        print(f"  ❌ Missing file: {e}")
        return False

    # Extract results by method
    agent_results = [
        r for r in agent_data.get("results", []) if r.get("method") == "agent"
    ]
    base_results = [
        r for r in base_data.get("results", []) if r.get("method") == "baseline"
    ]

    agent_metrics = calculate_metrics(agent_results)
    base_metrics = calculate_metrics(base_results)

    # Determine language pair from file
    source_lang = agent_data.get("source_lang", "unknown")
    target_lang = agent_data.get("target_lang", "unknown")
    lang_pair_display = f"{source_lang} → {target_lang}"

    # Build report
    report_lines = []
    report_lines.append("# Translation Quality Comparison Report")
    report_lines.append(f"## {lang_pair_display}: Agent vs Baseline")
    report_lines.append("")

    # Basic Info
    report_lines.append("### Basic Information")
    report_lines.append(f"- **Language Pair**: {lang_pair_display}")
    report_lines.append(f"- **Agent Tests Completed**: {len(agent_results)}")
    report_lines.append(f"- **Base Tests Completed**: {len(base_results)}")
    report_lines.append(f"- **Model**: qwen3:30b-a3b-instruct-2507-q4_K_M")
    report_lines.append(
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append("")

    # Warning if incomplete
    total_expected = max(len(agent_results), len(base_results))
    if len(agent_results) < total_expected:
        report_lines.append("### ⚠️ Data Completeness Warning")
        report_lines.append(
            f"- **Agent**: Only {len(agent_results)}/{total_expected} tests completed"
        )
        report_lines.append("")

    # Summary Table
    report_lines.append("### Performance Metrics Summary")
    report_lines.append("")
    report_lines.append("| Metric | Agent | Base | Difference | Winner |")
    report_lines.append("|--------|-------|------|-----------|--------|")

    ser_diff = agent_metrics["ser"] - base_metrics["ser"]
    ser_winner = "🟢 Agent" if agent_metrics["ser"] < base_metrics["ser"] else "🔴 Base"
    report_lines.append(
        f"| **SER** (↓) | {agent_metrics['ser']:.4f} | {base_metrics['ser']:.4f} | {ser_diff:+.4f} | {ser_winner} |"
    )

    scre_diff = agent_metrics["scre"] - base_metrics["scre"]
    scre_winner = (
        "🟢 Agent" if agent_metrics["scre"] < base_metrics["scre"] else "🔴 Base"
    )
    if agent_metrics["scre"] > 0:
        scre_ratio = base_metrics["scre"] / agent_metrics["scre"]
    else:
        scre_ratio = 0
    report_lines.append(
        f"| **SCRE** (↓) | {agent_metrics['scre']:.4f} | {base_metrics['scre']:.4f} | {scre_diff:+.4f} ({scre_ratio:.1f}x) | {scre_winner} |"
    )

    ari_diff = agent_metrics["ari"] - base_metrics["ari"]
    ari_winner = "🟢 Agent" if agent_metrics["ari"] > base_metrics["ari"] else "🔴 Base"
    report_lines.append(
        f"| **ARI** (↑) | {agent_metrics['ari']:.4f} | {base_metrics['ari']:.4f} | {ari_diff:+.4f} | {ari_winner} |"
    )

    time_diff = agent_metrics["avg_time_seconds"] - base_metrics["avg_time_seconds"]
    if base_metrics["avg_time_seconds"] > 0:
        speedup = agent_metrics["avg_time_seconds"] / base_metrics["avg_time_seconds"]
    else:
        speedup = 0
    time_winner = (
        "🟢 Agent"
        if agent_metrics["avg_time_seconds"] < base_metrics["avg_time_seconds"]
        else "🔴 Base"
    )
    report_lines.append(
        f"| **Time(s)** (↓) | {agent_metrics['avg_time_seconds']:.2f} | {base_metrics['avg_time_seconds']:.2f} | {time_diff:+.2f}s ({speedup:.1f}x) | {time_winner} |"
    )

    report_lines.append("")

    # Count wins
    agent_wins = sum(
        [
            agent_metrics["ser"] < base_metrics["ser"],
            agent_metrics["scre"] < base_metrics["scre"],
            agent_metrics["ari"] > base_metrics["ari"],
        ]
    )

    # Conclusion
    report_lines.append("### Conclusion")
    report_lines.append("")
    report_lines.append(f"**Agent wins on {agent_wins}/3 quality metrics**")
    report_lines.append("")

    # Critical warnings
    if base_metrics["scre"] > 3:
        report_lines.append("🚨 **CRITICAL**: Baseline is BROKEN (SCRE > 3)")
        report_lines.append("")

    if len(agent_results) < total_expected:
        report_lines.append(
            f"⚠️ **Incomplete**: Agent only completed {len(agent_results)}/{total_expected} tests"
        )
        report_lines.append("")

    # Recommendation
    if base_metrics["scre"] > 3:
        report_lines.append("**Recommendation**: Use Agent (Baseline is broken)")
    elif agent_wins >= 2 and speedup <= 2:
        report_lines.append(
            "**Recommendation**: Use Agent (Better quality, acceptable speed)"
        )
    elif agent_wins >= 2:
        report_lines.append(
            "**Recommendation**: Use Agent (Better quality despite slow speed)"
        )
    elif agent_wins == 1:
        report_lines.append("**Recommendation**: Use Base (Faster with mixed quality)")
    else:
        report_lines.append("**Recommendation**: Use Base (Better overall)")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*Generated: {datetime.now().isoformat()}*")

    # Save report
    report_content = "\n".join(report_lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return True


def main():
    output_dir = Path("benchmarks/results")
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # All language pairs
    pairs = [
        ("cmn→en-us", "cmn_en-us_comparison"),
        ("cmn→ja", "cmn_ja_comparison"),
        ("en-us→cmn", "en-us_cmn_comparison"),
        ("en-us→ja", "en-us_ja_comparison"),
        ("ja→cmn", "ja_cmn_comparison"),
        ("ja→en-us", "ja_en-us_comparison"),
    ]

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORTS")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0

    for pair, filename in pairs:
        print(f"📄 {pair}...", end=" ")

        agent_file = output_dir / f"{pair}_agent.json"
        base_file = output_dir / f"{pair}_base.json"
        report_path = report_dir / f"{filename}.md"

        if generate_report(agent_file, base_file, report_path):
            print(f"✅ Saved to {filename}.md")
            successful += 1
        else:
            print(f"❌ Failed")
            failed += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print(f"Reports saved to: {report_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
