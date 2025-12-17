"""
Comparison Reporter

Generates clean reports comparing agent vs baseline translation quality.
"""

from __future__ import annotations
from pathlib import Path
from .runner import ExperimentResults, TranslationResult


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
        lines.append(f"# Translation Quality Comparison Report")
        lines.append(f"")
        lines.append(f"**Experiment ID**: `{results.experiment_id}`")
        lines.append(f"**Date**: {results.timestamp}")
        lines.append(f"**Model**: {results.model}")
        lines.append(f"**Language Pair**: {results.language_pair}")
        lines.append(f"**Test Cases**: {results.total_tests}")
        lines.append(f"")

        # Executive Summary
        lines.append(f"## Executive Summary")
        lines.append(f"")
        lines.append(self._format_summary_table(results))
        lines.append(f"")

        # Detailed Metrics Comparison
        lines.append(f"## Detailed Metrics")
        lines.append(f"")
        lines.append(self._format_detailed_metrics(results))
        lines.append(f"")

        # Individual Test Results
        lines.append(f"## Individual Test Results")
        lines.append(f"")
        lines.append(self._format_individual_results(results))
        lines.append(f"")

        # Sample Translations
        lines.append(f"## Sample Translations")
        lines.append(f"")
        lines.append(self._format_sample_translations(results))
        lines.append(f"")

        # Analysis & Insights
        lines.append(f"## Analysis")
        lines.append(f"")
        lines.append(self._generate_analysis(results))
        lines.append(f"")

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
        agent = results.agent_avg_metrics
        baseline = results.baseline_avg_metrics

        def delta_percentage(metric):
            """Calculate delta as percentage (lower better for SER/SCRE)"""
            diff = agent[metric] - baseline[metric]
            if metric in ("ser", "scre"):
                # For SER and SCRE, lower is better, so negative delta is good
                if diff < -0.05:
                    indicator = "🟢"
                elif diff > 0.05:
                    indicator = "🔴"
                else:
                    indicator = "⚪"
                return f"{indicator} {diff:+.2f}" if metric == "ser" else f"{indicator} {diff:+.2%}"
            else:
                # For RPR, higher is better
                if diff > 0.05:
                    indicator = "🟢"
                elif diff < -0.05:
                    indicator = "🔴"
                else:
                    indicator = "⚪"
                return f"{indicator} {diff:+.2%}"

        lines = [
            "| Metric | Agent | Baseline | Delta |",
            "|--------|-------|----------|-------|",
            f"| **SER (Syllable Error Rate)** ↓ | {agent['ser']:.2f} | {baseline['ser']:.2f} | {delta_percentage('ser')} |",
            f"| **SCRE (Syllable Count Rel. Error)** ↓ | {agent['scre']:.2%} | {baseline['scre']:.2%} | {delta_percentage('scre')} |",
            f"| **RPR (Rhyme Preservation Rate)** ↑ | {agent['rpr']:.2%} | {baseline['rpr']:.2%} | {delta_percentage('rpr')} |",
            f"| **Avg Time (s)** | {agent['avg_time_seconds']:.1f} | {baseline['avg_time_seconds']:.1f} | {agent['avg_time_seconds'] - baseline['avg_time_seconds']:+.1f} |",
        ]

        return "\n".join(lines)

    def _format_detailed_metrics(self, results: ExperimentResults) -> str:
        """Format detailed metrics breakdown for three core metrics"""
        lines = []

        # SER (Syllable Error Rate)
        lines.append("### SER: Syllable Error Rate ↓")
        lines.append("")
        lines.append("*Lower is better. Measures edit distance between syllable sequences.*")
        lines.append("")
        lines.append(f"- **Agent SER**: {results.agent_avg_metrics['ser']:.3f}")
        lines.append(f"- **Baseline SER**: {results.baseline_avg_metrics['ser']:.3f}")
        ser_improvement = results.baseline_avg_metrics['ser'] - results.agent_avg_metrics['ser']
        lines.append(f"- **Improvement**: {ser_improvement:+.3f} (lower is better)")
        lines.append("")

        # SCRE (Syllable Count Relative Error)
        lines.append("### SCRE: Syllable Count Relative Error ↓")
        lines.append("")
        lines.append("*Lower is better. Average relative error in syllable counts per line.*")
        lines.append("")
        lines.append(f"- **Agent SCRE**: {results.agent_avg_metrics['scre']:.2%}")
        lines.append(f"- **Baseline SCRE**: {results.baseline_avg_metrics['scre']:.2%}")
        scre_improvement = results.baseline_avg_metrics['scre'] - results.agent_avg_metrics['scre']
        lines.append(f"- **Improvement**: {scre_improvement:+.2%} (lower is better)")
        lines.append("")

        # RPR (Rhyme Preservation Rate)
        lines.append("### RPR: Rhyme Preservation Rate ↑")
        lines.append("")
        lines.append("*Higher is better [0-1]. Fraction of target end-of-line rhymes preserved.*")
        lines.append("")
        lines.append(f"- **Agent RPR**: {results.agent_avg_metrics['rpr']:.2%}")
        lines.append(f"- **Baseline RPR**: {results.baseline_avg_metrics['rpr']:.2%}")
        rpr_improvement = results.agent_avg_metrics['rpr'] - results.baseline_avg_metrics['rpr']
        lines.append(f"- **Improvement**: {rpr_improvement:+.2%}")
        lines.append("")

        return "\n".join(lines)

    def _format_individual_results(self, results: ExperimentResults) -> str:
        """Format individual test results table for three core metrics"""
        lines = [
            "| Test ID | Method | SER ↓ | SCRE ↓ | RPR ↑ | Time (s) |",
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
                    f"{result.metrics['ser']:.2f} | "
                    f"{result.metrics['scre']:.2%} | "
                    f"{result.metrics['rpr']:.2%} | "
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

            baseline_result = next((r for r in test_results if r.method == "baseline"), None)
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
            lines.append(f"*Metrics*: SER={baseline_result.metrics['ser']:.2f}, "
                        f"SCRE={baseline_result.metrics['scre']:.2%}, "
                        f"RPR={baseline_result.metrics['rpr']:.2%}")
            lines.append("")

            # Agent
            lines.append(f"**Agent Translation ({agent_result.target_lang})**:")
            lines.append("```")
            for line in agent_result.translated_lines:
                lines.append(line)
            lines.append("```")
            lines.append(f"*Metrics*: SER={agent_result.metrics['ser']:.2f}, "
                        f"SCRE={agent_result.metrics['scre']:.2%}, "
                        f"RPR={agent_result.metrics['rpr']:.2%}")
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
        ser_improvement = baseline['ser'] - agent['ser']
        if ser_improvement > 0.1:
            lines.append(f"- ✅ **Strong syllable edit distance**: Agent reduces SER by {ser_improvement:.3f}")
        elif ser_improvement > 0.02:
            lines.append(f"- ✓ **Improved syllable accuracy**: Agent reduces SER by {ser_improvement:.3f}")
        else:
            lines.append(f"- ⚠️ **Limited syllable improvement**: Only {ser_improvement:.3f} difference in SER")

        # SCRE (Syllable Count Relative Error) - lower is better
        scre_improvement = baseline['scre'] - agent['scre']
        if scre_improvement > 0.1:
            lines.append(f"- ✅ **Strong relative error reduction**: Agent reduces SCRE by {scre_improvement:.2%}")
        elif scre_improvement > 0.05:
            lines.append(f"- ✓ **Improved relative accuracy**: Agent reduces SCRE by {scre_improvement:.2%}")
        else:
            lines.append(f"- ⚠️ **Limited relative improvement**: Only {scre_improvement:.2%} difference in SCRE")

        # RPR (Rhyme Preservation Rate) - higher is better
        rpr_improvement = agent['rpr'] - baseline['rpr']
        if rpr_improvement > 0.1:
            lines.append(f"- ✅ **Strong rhyme preservation**: Agent improves RPR by {rpr_improvement:.2%}")
        elif rpr_improvement > 0.05:
            lines.append(f"- ✓ **Improved rhyme matching**: Agent improves RPR by {rpr_improvement:.2%}")
        else:
            lines.append(f"- ⚠️ **Limited rhyme improvement**: Only {rpr_improvement:.2%} difference in RPR")

        # Time trade-off
        time_overhead = agent['avg_time_seconds'] - baseline['avg_time_seconds']
        if time_overhead > 0:
            lines.append(f"- ⏱️ **Time overhead**: Agent takes {time_overhead:.1f}s longer per translation ({time_overhead / baseline['avg_time_seconds']:.1%} increase)")
        else:
            lines.append(f"- ⚡ **Time advantage**: Agent is {-time_overhead:.1f}s faster per translation ({-time_overhead / baseline['avg_time_seconds']:.1%} faster)")

        lines.append("")

        # Recommendations
        lines.append("### Recommendations")
        lines.append("")

        # Overall score: count improvements
        improvements = sum([
            ser_improvement > 0.02,
            scre_improvement > 0.05,
            rpr_improvement > 0.05,
        ])

        if improvements >= 3:
            lines.append(f"- **Use agent for production**: Agent improves all three metrics")
        elif improvements >= 2:
            lines.append(f"- **Agent shows promise**: Consider using for quality-critical translations ({improvements}/3 metrics improved)")
        else:
            lines.append(f"- **Further optimization needed**: Agent improves only {improvements}/3 metrics")

        return "\n".join(lines)
