#!/usr/bin/env python3
"""
Run Translation Quality Experiments

CLI tool to benchmark translation quality comparing agent vs baseline.

Usage:
    # Run experiment for a single language pair
    python -m benchmarks.run_experiment cmn en-us --samples 5

    # Run all language pairs
    python -m benchmarks.run_experiment --all --samples 10

    # Use custom test suite
    python -m benchmarks.run_experiment --test-suite benchmarks/test_suites/cmn_en.json
"""

import os
import argparse
from pathlib import Path

# Disable LangChain tracing before any imports
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

from benchmarks.experiments import (
    ExperimentRunner,
    ComparisonReporter,
    sample_test_cases,
    load_lyrics_from_json,
    load_test_suite,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run translation quality experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Language pair arguments
    parser.add_argument(
        "source_lang",
        nargs="?",
        help="Source language (cmn, en-us, ja)",
    )
    parser.add_argument(
        "target_lang",
        nargs="?",
        help="Target language (cmn, en-us, ja)",
    )

    # Options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all language pairs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of test samples per language pair (default: 10)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to pre-created test suite JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/data",
        help="Directory with scraped lyrics data (default: benchmarks/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=20,
        help="Maximum lines per test case (default: 20)",
    )

    args = parser.parse_args()

    # Validate arguments
    if (
        not args.all
        and not args.test_suite
        and not (args.source_lang and args.target_lang)
    ):
        parser.error("Must specify language pair, --all, or --test-suite")

    # Initialize runner and reporter
    print("\n🔬 Initializing experiment runner...")
    print(f"   Model: {args.model}")
    print(f"   Ollama URL: {args.base_url}")

    runner = ExperimentRunner(model=args.model, base_url=args.base_url)
    reporter = ComparisonReporter()

    # Determine language pairs to test
    if args.test_suite:
        # Load pre-created test suite
        print(f"\n📂 Loading test suite: {args.test_suite}")
        test_cases = load_test_suite(args.test_suite)
        language_pairs = [(test_cases[0]["source_lang"], test_cases[0]["target_lang"])]

    elif args.all:
        # Run all pairs
        language_pairs = [
            ("cmn", "en-us"),
            ("cmn", "ja"),
            ("en-us", "cmn"),
            ("en-us", "ja"),
            ("ja", "cmn"),
            ("ja", "en-us"),
        ]
        print(
            f"\n🌐 Running experiments for all language pairs ({len(language_pairs)} pairs)"
        )

    else:
        # Single pair
        language_pairs = [(args.source_lang, args.target_lang)]
        print(f"\n🔀 Running experiment: {args.source_lang} → {args.target_lang}")

    # Run experiments for each pair
    all_results = []
    failed_pairs = []

    for source_lang, target_lang in language_pairs:
        pair_key = f"{source_lang}→{target_lang}"
        print(f"\n{'=' * 60}")
        print(f"Language Pair: {pair_key}")
        print(f"{'=' * 60}")

        try:
            # Create test cases if not using pre-created suite
            if args.test_suite:
                # Filter test cases for this pair
                pair_test_cases = [
                    tc
                    for tc in test_cases
                    if tc["source_lang"] == source_lang
                    and tc["target_lang"] == target_lang
                ]
            else:
                # Sample from scraped data
                print(f"\n📊 Sampling {args.samples} test cases...")

                # Determine source file
                source_file_map = {
                    "cmn": "cmn_lyrics.json",
                    "en-us": "en_lyrics.json",
                    "ja": "ja_lyrics.json",
                }

                source_file = Path(args.data_dir) / source_file_map.get(
                    source_lang, f"{source_lang}_lyrics.json"
                )

                if not source_file.exists():
                    print(f"❌ Error: Source data file not found: {source_file}")
                    continue

                source_lyrics = load_lyrics_from_json(source_file, source_lang)
                pair_test_cases = sample_test_cases(
                    source_lyrics=source_lyrics,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    num_samples=args.samples,
                    max_lines=args.max_lines,
                )

            if not pair_test_cases:
                print(f"⚠️  No test cases for {pair_key}, skipping")
                continue

            print(f"   Test cases: {len(pair_test_cases)}")

            # Run experiment
            output_dir = Path(args.output_dir)
            results = runner.run_experiment(
                test_cases=pair_test_cases,
                experiment_id=f"{pair_key}_{len(pair_test_cases)}samples",
            )

            all_results.append(results)

            # Save final results
            runner.save_results(results, output_dir)

            # Generate and save report
            report_path = output_dir / f"{results.experiment_id}.md"
            reporter.generate_markdown_report(results, report_path)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"\n❌ Error processing {pair_key}: {error_msg}")
            failed_pairs.append((pair_key, error_msg))
            print("⏭️  Continuing with next language pair...")
            continue

        # Print summary
        print(f"\n📊 Summary for {pair_key}:")

        # SER (lower is better)
        ser_agent = results.agent_avg_metrics["ser"]
        ser_baseline = results.baseline_avg_metrics["ser"]
        print("   SER (Syllable Error Rate) ↓")
        print(f"      Agent:      {ser_agent:.2%}")
        print(f"      Baseline:   {ser_baseline:.2%}")
        print(f"      Improvement: {ser_baseline - ser_agent:+.2%}")

        # SCRE (lower is better)
        scre_agent = results.agent_avg_metrics["scre"]
        scre_baseline = results.baseline_avg_metrics["scre"]
        print("   SCRE (Syllable Count Relative Error) ↓")
        print(f"      Agent:      {scre_agent:.2%}")
        print(f"      Baseline:   {scre_baseline:.2%}")
        print(f"      Improvement: {scre_baseline - scre_agent:+.2%}")

        # ARI (higher is better, range [-1, 1])
        ari_agent = results.agent_avg_metrics["ari"]
        ari_baseline = results.baseline_avg_metrics["ari"]
        print("   ARI (Adjusted Rand Index) ↑")
        print(f"      Agent:      {ari_agent:.3f}")
        print(f"      Baseline:   {ari_baseline:.3f}")
        print(f"      Improvement: {ari_agent - ari_baseline:+.3f}")

    # Final summary
    print(f"\n{'=' * 60}")
    if failed_pairs:
        print(f"⚠️  Experiments Complete (with {len(failed_pairs)} failures)")
    else:
        print("✅ Experiments Complete!")
    print(f"{'=' * 60}")
    print(f"   Language pairs completed: {len(all_results)}")
    if failed_pairs:
        print(f"   Language pairs failed: {len(failed_pairs)}")
        print()
        for pair_key, error in failed_pairs:
            print(f"     ❌ {pair_key}: {error}")
    print(f"   Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
