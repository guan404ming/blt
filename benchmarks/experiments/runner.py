"""
Experiment Runner

Orchestrates translation experiments comparing agent vs baseline.
"""

from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import TypedDict, Literal
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .baseline import BaselineTranslator
from .evaluator import TranslationEvaluator, EvaluationMetrics

# Disable LangChain tracing to prevent memory allocation errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"


class TestCase(TypedDict):
    """Single test case for translation"""

    id: str
    source_lines: list[str]
    source_lang: str
    target_lang: str
    metadata: dict  # Genre, artist, etc.


@dataclass
class TranslationResult:
    """Result from a single translation"""

    test_id: str
    source_lang: str
    target_lang: str
    method: Literal["agent", "baseline"]  # Which translation method

    # Translation
    source_lines: list[str]
    translated_lines: list[str]

    # Metrics
    metrics: EvaluationMetrics

    # Performance
    time_seconds: float

    # Metadata
    metadata: dict


@dataclass
class ExperimentResults:
    """Results from full experiment run"""

    experiment_id: str
    timestamp: str
    model: str
    language_pair: str  # e.g., "cmn→en"

    # Results
    results: list[TranslationResult]

    # Summary stats
    total_tests: int
    agent_avg_metrics: dict
    baseline_avg_metrics: dict


class ExperimentRunner:
    """
    Runs translation experiments comparing agent vs baseline

    For each test case:
    1. Translate with agent (constraint-aware)
    2. Translate with baseline (standard NMT)
    3. Evaluate both translations
    4. Collect metrics for comparison
    """

    def __init__(
        self,
        model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize experiment runner

        Args:
            model: Ollama model name
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url

        # Ensure LangChain tracing is disabled
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

        # Initialize components (lazy imports to avoid circular dependencies)
        from blt.translators import (
            LyricsAnalyzer,
            LyricsTranslationAgent,
            LyricsTranslationAgentConfig,
        )

        self.analyzer = LyricsAnalyzer()
        self.evaluator = TranslationEvaluator(self.analyzer)
        self.baseline = BaselineTranslator(model=model, base_url=base_url)

        # Configure agent
        config = LyricsTranslationAgentConfig(
            model=model,
            ollama_base_url=base_url,
        )
        self.agent = LyricsTranslationAgent(
            config=config,
            analyzer=self.analyzer,
        )

    def run_experiment(
        self,
        test_cases: list[TestCase],
        experiment_id: str | None = None,
        mode: Literal["base", "agent"] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 5,
    ) -> ExperimentResults:
        """
        Run full experiment on test cases with checkpoint support

        Args:
            test_cases: List of test cases to evaluate
            experiment_id: Optional experiment ID (auto-generated if None)
            mode: Run only "base" (baseline) or "agent" (None = run both)
            checkpoint_path: Path to save/resume checkpoints
            checkpoint_interval: Save checkpoint every N test cases (default: 5)

        Returns:
            ExperimentResults with all results and comparisons
        """
        if not test_cases:
            raise ValueError("No test cases provided")

        # Convert checkpoint_path to Path
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)

        # Generate experiment ID
        if experiment_id is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_pair = f"{test_cases[0]['source_lang']}→{test_cases[0]['target_lang']}"
            experiment_id = f"{lang_pair}_{timestamp}"
        else:
            timestamp = experiment_id

        # Try to load existing checkpoint
        start_index = 0
        results: list[TranslationResult] = []
        failed_tests: list[tuple[str, str]] = []

        if checkpoint_path and checkpoint_path.exists():
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    # Reconstruct results from checkpoint
                    for result_dict in checkpoint_data.get("results", []):
                        result_dict["metrics"] = result_dict.get("metrics", {})
                        result = TranslationResult(
                            test_id=result_dict["test_id"],
                            source_lang=result_dict["source_lang"],
                            target_lang=result_dict["target_lang"],
                            method=result_dict["method"],
                            source_lines=result_dict["source_lines"],
                            translated_lines=result_dict["translated_lines"],
                            metrics=result_dict["metrics"],
                            time_seconds=result_dict["time_seconds"],
                            metadata=result_dict.get("metadata", {}),
                        )
                        results.append(result)
                    # Calculate how many tests have been completed
                    completed_test_ids = set(r.test_id for r in results)
                    for idx, tc in enumerate(test_cases):
                        if tc["id"] not in completed_test_ids:
                            start_index = idx
                            break
                    else:
                        # All tests already completed
                        start_index = len(test_cases)
                    print(f"✅ Resumed from test {start_index + 1}/{len(test_cases)}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                results = []
                start_index = 0

        # Run test cases starting from start_index
        test_iter = enumerate(test_cases[start_index:], start=start_index + 1)
        test_iter = tqdm(
            test_iter,
            total=len(test_cases),
            initial=start_index,
            desc=f"Tests [{test_cases[0]['source_lang']}→{test_cases[0]['target_lang']}]",
            unit="test",
            leave=False,
        )

        for i, test_case in test_iter:
            # print(f"\n[{i}/{len(test_cases)}] Running test: {test_case['id']}")
            # print(
            #     f"  Language pair: {test_case['source_lang']} → {test_case['target_lang']}"
            # )

            try:
                # Run baseline translation
                if mode is None or mode == "base":
                    print("  - Baseline translation...")
                    baseline_result = self._run_baseline(test_case)
                    results.append(baseline_result)
                else:
                    baseline_result = None

                # Run agent translation
                if mode is None or mode == "agent":
                    print("  - Agent translation...")
                    agent_result = self._run_agent(test_case)
                    results.append(agent_result)
                else:
                    agent_result = None

                print("\n  Results:")
                print(f"  Source Line 1 - 2: {test_case['source_lines'][0:2]}")
                if baseline_result:
                    print(
                        f"  Baseline Line 1 - 2: {baseline_result.translated_lines[0:2] if baseline_result.translated_lines else 'N/A'}"
                    )
                if agent_result:
                    print(
                        f"  Agent Line 1 - 2:    {agent_result.translated_lines[0:2] if agent_result.translated_lines else 'N/A'}"
                    )
                print()
                if baseline_result:
                    print(
                        f"    Baseline - SER: {baseline_result.metrics['ser']:.2%}, "
                        f"SCRE: {baseline_result.metrics['scre']:.2%}, "
                        f"ARI: {baseline_result.metrics['ari']:.2f}"
                    )
                if agent_result:
                    print(
                        f"    Agent    - SER: {agent_result.metrics['ser']:.2%}, "
                        f"SCRE: {agent_result.metrics['scre']:.2%}, "
                        f"ARI: {agent_result.metrics['ari']:.2f}"
                    )

                # Save checkpoint every N test cases
                if checkpoint_path and i % checkpoint_interval == 0:
                    self._save_checkpoint(
                        checkpoint_path,
                        results,
                        experiment_id,
                        test_cases[0]["source_lang"],
                        test_cases[0]["target_lang"],
                    )
                    print(f"💾 Checkpoint saved ({i} tests completed)")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                print(f"  ❌ Error: {error_msg}")
                failed_tests.append((test_case["id"], error_msg))
                print("  ⏭️  Continuing with next test...")

        # Print failed tests summary
        result_count = len(results) // (
            1 if mode else 2
        )  # Divide by 2 only if running both
        if failed_tests:
            print(f"\n{'=' * 60}")
            print(f"⚠️  {len(failed_tests)} test(s) failed:")
            print(f"{'=' * 60}")
            for test_id, error in failed_tests:
                print(f"  ❌ {test_id}: {error}")
            print(f"✅ {result_count} test(s) completed successfully")
        else:
            print(f"\n{'=' * 60}")
            print(f"✅ All {len(test_cases)} test(s) completed successfully!")
            print(f"{'=' * 60}")

        # Calculate summary statistics
        lang_pair = f"{test_cases[0]['source_lang']}→{test_cases[0]['target_lang']}"

        agent_results = [r for r in results if r.method == "agent"]
        baseline_results = [r for r in results if r.method == "baseline"]

        agent_avg = self._calculate_average_metrics(agent_results)
        baseline_avg = self._calculate_average_metrics(baseline_results)

        return ExperimentResults(
            experiment_id=experiment_id,
            timestamp=timestamp,
            model=self.model,
            language_pair=lang_pair,
            results=results,
            total_tests=len(test_cases),
            agent_avg_metrics=agent_avg,
            baseline_avg_metrics=baseline_avg,
        )

    def _run_baseline(self, test_case: TestCase) -> TranslationResult:
        """Run baseline translation"""
        start_time = time.time()

        translation = self.baseline.translate(
            source_lines=test_case["source_lines"],
            source_lang=test_case["source_lang"],
            target_lang=test_case["target_lang"],
        )

        elapsed = time.time() - start_time

        # Evaluate
        metrics = self.evaluator.evaluate(
            source_lines=test_case["source_lines"],
            source_lang=test_case["source_lang"],
            translated_lines=translation["translated_lines"],
            target_lang=test_case["target_lang"],
        )

        return TranslationResult(
            test_id=test_case["id"],
            source_lang=test_case["source_lang"],
            target_lang=test_case["target_lang"],
            method="baseline",
            source_lines=test_case["source_lines"],
            translated_lines=translation["translated_lines"],
            metrics=metrics,
            time_seconds=elapsed,
            metadata=test_case["metadata"],
        )

    def _run_agent(self, test_case: TestCase) -> TranslationResult:
        """Run agent translation"""
        start_time = time.time()

        # Join lines for agent (it expects single string)
        source_text = "\n".join(test_case["source_lines"])

        translation = self.agent.translate(
            source_lyrics=source_text,
            source_lang=test_case["source_lang"],
            target_lang=test_case["target_lang"],
        )

        elapsed = time.time() - start_time

        # Evaluate
        metrics = self.evaluator.evaluate(
            source_lines=test_case["source_lines"],
            source_lang=test_case["source_lang"],
            translated_lines=translation.translated_lines,
            target_lang=test_case["target_lang"],
        )

        return TranslationResult(
            test_id=test_case["id"],
            source_lang=test_case["source_lang"],
            target_lang=test_case["target_lang"],
            method="agent",
            source_lines=test_case["source_lines"],
            translated_lines=translation.translated_lines,
            metrics=metrics,
            time_seconds=elapsed,
            metadata=test_case["metadata"],
        )

    def _calculate_average_metrics(
        self,
        results: list[TranslationResult],
    ) -> dict:
        """Calculate average metrics across results (SER, SCRE, ARI)"""
        if not results:
            return {}

        # Three core metrics
        metrics_keys = ["ser", "scre", "ari"]

        averages = {}
        for key in metrics_keys:
            values = [
                r.metrics.get(key) for r in results if r.metrics.get(key) is not None
            ]
            if values:
                averages[key] = sum(values) / len(values)

        # Average time
        if results:
            averages["avg_time_seconds"] = sum(r.time_seconds for r in results) / len(
                results
            )

        return averages

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        results: list[TranslationResult],
        experiment_id: str,
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Save intermediate checkpoint"""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to dict
        results_dict = {
            "experiment_id": experiment_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "results": [asdict(r) for r in results],
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    def save_results(
        self,
        results: ExperimentResults,
        output_dir: str | Path = "benchmarks/results",
    ) -> Path:
        """
        Save experiment results to JSON

        Args:
            results: Experiment results to save
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{results.experiment_id}.json"
        filepath = output_path / filename

        # Convert to dict (handle dataclasses)
        results_dict = {
            "experiment_id": results.experiment_id,
            "timestamp": results.timestamp,
            "model": results.model,
            "language_pair": results.language_pair,
            "total_tests": results.total_tests,
            "agent_avg_metrics": results.agent_avg_metrics,
            "baseline_avg_metrics": results.baseline_avg_metrics,
            "results": [asdict(r) for r in results.results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Results saved to: {filepath}")
        return filepath
