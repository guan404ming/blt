"""
Comparison Reporter

Generates clean reports comparing agent vs baseline translation quality.
"""

from __future__ import annotations
from pathlib import Path
from .runner import ExperimentResults


class ComparisonReporter:
    """
    Generates comparison reports for experiment results

    Produces:
    - Markdown reports (human-readable)
    - JSON summaries (machine-readable)
    - Side-by-side translation comparisons
    """

    def generate_markdown_report(
        self,
        results: ExperimentResults,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate comprehensive markdown report

        Args:
            results: Experiment results
            output_path: Optional path to save report

        Returns:
            Markdown report content
        """
        lines = []

        # Header
        lines.append("# Translation Quality Comparison Report")
        lines.append("")
        lines.append(f"**Experiment ID**: `{results.experiment_id}`")
        lines.append(f"**Date**: {results.timestamp}")
        lines.append(f"**Model**: {results.model}")
        lines.append(f"**Language Pair**: {results.language_pair}")
        lines.append(f"**Test Cases**: {results.total_tests}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self._format_summary_table(results))
        lines.append("")

        # Detailed Metrics Comparison
        lines.append("## Detailed Metrics")
        lines.append("")
        lines.append(self._format_detailed_metrics(results))
        lines.append("")

        # Individual Test Results
        lines.append("## Individual Test Results")
        lines.append("")
        lines.append(self._format_individual_results(results))
        lines.append("")

        # Sample Translations
        lines.append("## Sample Translations")
        lines.append("")
        lines.append(self._format_sample_translations(results))
        lines.append("")

        # Analysis & Insights
        lines.append("## Analysis")
        lines.append("")
        lines.append(self._generate_analysis(results))
        lines.append("")

        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"📝 Markdown report saved to: {output_path}")

        return report

    def _format_summary_table(self, results: ExperimentResults) -> str:
        """Format executive summary table for three core metrics"""
        agent = results.agent_avg_metrics or {}
        baseline = results.baseline_avg_metrics or {}

        # Check if we have both methods or just one
        has_agent = bool(agent)
        has_baseline = bool(baseline)

        # If only one method, show single column
        if has_agent and not has_baseline:
            lines = [
                "| Metric | Agent |",
                "|--------|-------|",
                f"| **SER (Syllable Error Rate)** ↓ | {agent.get('ser', 'N/A'):.2%} |"
                if agent.get("ser")
                else "| **SER (Syllable Error Rate)** ↓ | N/A |",
                f"| **SCRE (Syllable Count Rel. Error)** ↓ | {agent.get('scre', 'N/A'):.2%} |"
                if agent.get("scre")
                else "| **SCRE (Syllable Count Rel. Error)** ↓ | N/A |",
                f"| **ARI (Adjusted Rand Index)** ↑ | {agent.get('ari', 'N/A'):.2f} |"
                if agent.get("ari")
                else "| **ARI (Adjusted Rand Index)** ↑ | N/A |",
                f"| **Avg Time (s)** | {agent.get('avg_time_seconds', 'N/A'):.1f} |"
                if agent.get("avg_time_seconds")
                else "| **Avg Time (s)** | N/A |",
            ]
        elif has_baseline and not has_agent:
            lines = [
                "| Metric | Baseline |",
                "|--------|----------|",
                f"| **SER (Syllable Error Rate)** ↓ | {baseline.get('ser', 'N/A'):.2%} |"
                if baseline.get("ser")
                else "| **SER (Syllable Error Rate)** ↓ | N/A |",
                f"| **SCRE (Syllable Count Rel. Error)** ↓ | {baseline.get('scre', 'N/A'):.2%} |"
                if baseline.get("scre")
                else "| **SCRE (Syllable Count Rel. Error)** ↓ | N/A |",
                f"| **ARI (Adjusted Rand Index)** ↑ | {baseline.get('ari', 'N/A'):.2f} |"
                if baseline.get("ari")
                else "| **ARI (Adjusted Rand Index)** ↑ | N/A |",
                f"| **Avg Time (s)** | {baseline.get('avg_time_seconds', 'N/A'):.1f} |"
                if baseline.get("avg_time_seconds")
                else "| **Avg Time (s)** | N/A |",
            ]
        else:
            # Both methods present - show comparison
            def delta_percentage(metric):
                """Calculate delta as percentage (lower better for SER/SCRE, higher for ARI)"""
                agent_val = agent.get(metric)
                baseline_val = baseline.get(metric)
                if agent_val is None or baseline_val is None:
                    return "N/A"

                diff = agent_val - baseline_val
                if metric in ("ser", "scre"):
                    # For SER and SCRE, lower is better, so negative delta is good
                    if diff < -0.05:
                        indicator = "🟢"
                    elif diff > 0.05:
                        indicator = "🔴"
                    else:
                        indicator = "⚪"
                    return f"{indicator} {diff:+.2%}"
                elif metric == "ari":
                    # For ARI, higher is better
                    if diff > 0.1:
                        indicator = "🟢"
                    elif diff < -0.1:
                        indicator = "🔴"
                    else:
                        indicator = "⚪"
                    return f"{indicator} {diff:+.2f}"
                else:
                    # Default: higher is better
                    if diff > 0.05:
                        indicator = "🟢"
                    elif diff < -0.05:
                        indicator = "🔴"
                    else:
                        indicator = "⚪"
                    return f"{indicator} {diff:+.2%}"

            ser_val = f"{agent.get('ser', 'N/A'):.2%}" if agent.get("ser") else "N/A"
            scre_val = f"{agent.get('scre', 'N/A'):.2%}" if agent.get("scre") else "N/A"
            ari_val = f"{agent.get('ari', 'N/A'):.2f}" if agent.get("ari") else "N/A"

            lines = [
                "| Metric | Agent | Baseline | Delta |",
                "|--------|-------|----------|-------|",
                f"| **SER (Syllable Error Rate)** ↓ | {ser_val} | {baseline.get('ser', 'N/A'):.2%} | {delta_percentage('ser')} |"
                if baseline.get("ser")
                else f"| **SER (Syllable Error Rate)** ↓ | {ser_val} | N/A | N/A |",
                f"| **SCRE (Syllable Count Rel. Error)** ↓ | {scre_val} | {baseline.get('scre', 'N/A'):.2%} | {delta_percentage('scre')} |"
                if baseline.get("scre")
                else f"| **SCRE (Syllable Count Rel. Error)** ↓ | {scre_val} | N/A | N/A |",
                f"| **ARI (Adjusted Rand Index)** ↑ | {ari_val} | {baseline.get('ari', 'N/A'):.2f} | {delta_percentage('ari')} |"
                if baseline.get("ari")
                else f"| **ARI (Adjusted Rand Index)** ↑ | {ari_val} | N/A | N/A |",
                f"| **Avg Time (s)** | {agent.get('avg_time_seconds', 'N/A'):.1f} | {baseline.get('avg_time_seconds', 'N/A'):.1f} | {agent.get('avg_time_seconds', 0) - baseline.get('avg_time_seconds', 0):+.1f} |"
                if (agent.get("avg_time_seconds") and baseline.get("avg_time_seconds"))
                else "| **Avg Time (s)** | N/A | N/A | N/A |",
            ]

        return "\n".join(lines)

    def _format_detailed_metrics(self, results: ExperimentResults) -> str:
        """Format detailed metrics breakdown for three core metrics"""
        lines = []

        agent = results.agent_avg_metrics or {}
        baseline = results.baseline_avg_metrics or {}

        # SER (Syllable Error Rate)
        if agent.get("ser") or baseline.get("ser"):
            lines.append("### SER: Syllable Error Rate ↓")
            lines.append("")
            lines.append(
                "*Lower is better. Measures edit distance between syllable sequences.*"
            )
            lines.append("")
            if agent.get("ser"):
                lines.append(f"- **Agent SER**: {agent['ser']:.2%}")
            if baseline.get("ser"):
                lines.append(f"- **Baseline SER**: {baseline['ser']:.2%}")
            if agent.get("ser") and baseline.get("ser"):
                ser_improvement = baseline["ser"] - agent["ser"]
                lines.append(
                    f"- **Improvement**: {ser_improvement:+.2%} (lower is better)"
                )
            lines.append("")

        # SCRE (Syllable Count Relative Error)
        if agent.get("scre") or baseline.get("scre"):
            lines.append("### SCRE: Syllable Count Relative Error ↓")
            lines.append("")
            lines.append(
                "*Lower is better. Average relative error in syllable counts per line.*"
            )
            lines.append("")
            if agent.get("scre"):
                lines.append(f"- **Agent SCRE**: {agent['scre']:.2%}")
            if baseline.get("scre"):
                lines.append(f"- **Baseline SCRE**: {baseline['scre']:.2%}")
            if agent.get("scre") and baseline.get("scre"):
                scre_improvement = baseline["scre"] - agent["scre"]
                lines.append(
                    f"- **Improvement**: {scre_improvement:+.2%} (lower is better)"
                )
            lines.append("")

        # ARI (Adjusted Rand Index)
        if agent.get("ari") is not None or baseline.get("ari") is not None:
            lines.append("### ARI: Adjusted Rand Index ↑")
            lines.append("")
            lines.append(
                "*Higher is better [-1, 1]. Measures rhyme clustering agreement.*"
            )
            lines.append("")
            if agent.get("ari") is not None:
                lines.append(f"- **Agent ARI**: {agent['ari']:.3f}")
            if baseline.get("ari") is not None:
                lines.append(f"- **Baseline ARI**: {baseline['ari']:.3f}")
            if agent.get("ari") is not None and baseline.get("ari") is not None:
                ari_improvement = agent["ari"] - baseline["ari"]
                lines.append(f"- **Improvement**: {ari_improvement:+.3f}")
            lines.append("")

        return "\n".join(lines)

    def _format_individual_results(self, results: ExperimentResults) -> str:
        """Format individual test results table for three core metrics"""
        lines = [
            "| Test ID | Method | SER ↓ | SCRE ↓ | ARI ↑ | Time (s) |",
            "|---------|--------|-------|--------|-------|----------|",
        ]

        # Group by test ID
        test_ids = sorted(set(r.test_id for r in results.results))

        for test_id in test_ids:
            test_results = [r for r in results.results if r.test_id == test_id]

            for result in sorted(test_results, key=lambda r: r.method):
                method_icon = "🤖" if result.method == "agent" else "📝"
                lines.append(
                    f"| {test_id} | {method_icon} {result.method.title()} | "
                    f"{result.metrics['ser']:.2%} | "
                    f"{result.metrics['scre']:.2%} | "
                    f"{result.metrics['ari']:.2f} | "
                    f"{result.time_seconds:.1f} |"
                )

        return "\n".join(lines)

    def _format_sample_translations(self, results: ExperimentResults) -> str:
        """Format sample translations for inspection"""
        lines = []

        # Pick first 3 test cases
        test_ids = sorted(set(r.test_id for r in results.results))[:3]

        for i, test_id in enumerate(test_ids, 1):
            test_results = [r for r in results.results if r.test_id == test_id]
            if not test_results:
                continue

            baseline_result = next(
                (r for r in test_results if r.method == "baseline"), None
            )
            agent_result = next((r for r in test_results if r.method == "agent"), None)

            if not (baseline_result and agent_result):
                continue

            lines.append(f"### Sample {i}: {test_id}")
            lines.append("")

            # Source
            lines.append(f"**Source ({baseline_result.source_lang})**:")
            lines.append("```")
            for line in baseline_result.source_lines:
                lines.append(line)
            lines.append("```")
            lines.append("")

            # Baseline
            lines.append(f"**Baseline Translation ({baseline_result.target_lang})**:")
            lines.append("```")
            for line in baseline_result.translated_lines:
                lines.append(line)
            lines.append("```")
            lines.append(
                f"*Metrics*: SER={baseline_result.metrics['ser']:.2%}, "
                f"SCRE={baseline_result.metrics['scre']:.2%}, "
                f"ARI={baseline_result.metrics['ari']:.2f}"
            )
            lines.append("")

            # Agent
            lines.append(f"**Agent Translation ({agent_result.target_lang})**:")
            lines.append("```")
            for line in agent_result.translated_lines:
                lines.append(line)
            lines.append("```")
            lines.append(
                f"*Metrics*: SER={agent_result.metrics['ser']:.2%}, "
                f"SCRE={agent_result.metrics['scre']:.2%}, "
                f"ARI={agent_result.metrics['ari']:.2f}"
            )
            lines.append("")

        return "\n".join(lines)

    def _generate_analysis(self, results: ExperimentResults) -> str:
        """Generate analysis and insights for three core metrics"""
        lines = []

        agent = results.agent_avg_metrics
        baseline = results.baseline_avg_metrics

        # Key findings
        lines.append("### Key Findings")
        lines.append("")

        # SER (Syllable Error Rate) - lower is better
        ser_improvement = baseline["ser"] - agent["ser"]
        if ser_improvement > 0.1:
            lines.append(
                f"- ✅ **Strong syllable edit distance**: Agent reduces SER by {ser_improvement:.2%}"
            )
        elif ser_improvement > 0.02:
            lines.append(
                f"- ✓ **Improved syllable accuracy**: Agent reduces SER by {ser_improvement:.2%}"
            )
        else:
            lines.append(
                f"- ⚠️ **Limited syllable improvement**: Only {ser_improvement:.2%} difference in SER"
            )

        # SCRE (Syllable Count Relative Error) - lower is better
        scre_improvement = baseline["scre"] - agent["scre"]
        if scre_improvement > 0.1:
            lines.append(
                f"- ✅ **Strong relative error reduction**: Agent reduces SCRE by {scre_improvement:.2%}"
            )
        elif scre_improvement > 0.05:
            lines.append(
                f"- ✓ **Improved relative accuracy**: Agent reduces SCRE by {scre_improvement:.2%}"
            )
        else:
            lines.append(
                f"- ⚠️ **Limited relative improvement**: Only {scre_improvement:.2%} difference in SCRE"
            )

        # ARI (Adjusted Rand Index) - higher is better, range [-1, 1]
        ari_improvement = agent["ari"] - baseline["ari"]
        if ari_improvement > 0.1:
            lines.append(
                f"- ✅ **Strong rhyme clustering**: Agent improves ARI by {ari_improvement:+.3f}"
            )
        elif ari_improvement > 0.05:
            lines.append(
                f"- ✓ **Improved rhyme matching**: Agent improves ARI by {ari_improvement:+.3f}"
            )
        else:
            lines.append(
                f"- ⚠️ **Limited rhyme improvement**: Only {ari_improvement:+.3f} difference in ARI"
            )

        # Time trade-off
        time_overhead = agent["avg_time_seconds"] - baseline["avg_time_seconds"]
        if time_overhead > 0:
            lines.append(
                f"- ⏱️ **Time overhead**: Agent takes {time_overhead:.1f}s longer per translation ({time_overhead / baseline['avg_time_seconds']:.1%} increase)"
            )
        else:
            lines.append(
                f"- ⚡ **Time advantage**: Agent is {-time_overhead:.1f}s faster per translation ({-time_overhead / baseline['avg_time_seconds']:.1%} faster)"
            )

        lines.append("")

        # Recommendations
        lines.append("### Recommendations")
        lines.append("")

        # Overall score: count improvements
        improvements = sum(
            [
                ser_improvement > 0.02,
                scre_improvement > 0.05,
                ari_improvement > 0.05,
            ]
        )

        if improvements >= 3:
            lines.append(
                "- **Use agent for production**: Agent improves all three metrics"
            )
        elif improvements >= 2:
            lines.append(
                f"- **Agent shows promise**: Consider using for quality-critical translations ({improvements}/3 metrics improved)"
            )
        else:
            lines.append(
                f"- **Further optimization needed**: Agent improves only {improvements}/3 metrics"
            )

        return "\n".join(lines)
