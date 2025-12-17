#!/usr/bin/env python3
"""
Run Translation Quality Experiments

CLI tool to benchmark translation quality.

Usage:
    # Run agent for single language pair (load all from JSON)
    python -m benchmarks.run_experiment cmn en-us

    # Run agent with specific sample count
    python -m benchmarks.run_experiment cmn en-us --samples 5

    # Run baseline for single language pair
    python -m benchmarks.run_experiment cmn en-us --mode base

    # Run all language pairs
    python -m benchmarks.run_experiment --all

    # Use custom test suite
    python -m benchmarks.run_experiment --test-suite benchmarks/test_suites/cmn_en.json
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

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
        "-m",
        "--mode",
        choices=["base", "agent"],
        default="agent",
        help="Run base (baseline) or agent (default: agent)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples per pair (default: all from JSON)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to test suite JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/data",
        help="Lyrics data directory (default: benchmarks/data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=5,
        help="Max lines per test case (default: 5)",
    )

    args = parser.parse_args()

    # Validate arguments
    if (
        not args.all
        and not args.test_suite
        and not (args.source_lang and args.target_lang)
    ):
        parser.error("Must specify language pair, --all, or --test-suite")

    # Fixed values
    base_url = "http://localhost:11434"
    output_dir = Path("benchmarks/results")

    # Initialize runner and reporter
    print("\n🔬 Initializing experiment runner...")
    print(f"   Mode: {args.mode}")
    print(f"   Model: {args.model}")

    runner = ExperimentRunner(model=args.model, base_url=base_url)
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

    for source_lang, target_lang in tqdm(
        language_pairs, desc="Language Pairs", unit="pair"
    ):
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

                # If samples not specified, load all from JSON
                if args.samples is None:
                    print(f"\n📊 Loading all test cases from {source_file.name}...")
                    pair_test_cases = sample_test_cases(
                        source_lyrics=source_lyrics,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        num_samples=None,  # None means all
                        max_lines=args.max_lines,
                    )
                else:
                    print(f"\n📊 Sampling {args.samples} test cases...")
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

            # Set checkpoint path with mode
            checkpoint_path = output_dir / f"{pair_key}_{args.mode}.json"

            # Run experiment
            results = runner.run_experiment(
                test_cases=pair_test_cases,
                experiment_id=f"{pair_key}_{args.mode}",
                mode=args.mode,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=5,
            )

            all_results.append(results)

            # Save final results
            runner.save_results(results, output_dir)

            # Generate and save report
            try:
                report_path = output_dir / f"{results.experiment_id}.md"
                reporter.generate_markdown_report(results, report_path)
            except Exception as e:
                print(f"⚠️  Could not generate report: {e}")

            # Check if we have valid metrics
            has_metrics = bool(
                results.agent_avg_metrics or results.baseline_avg_metrics
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"\n❌ Error processing {pair_key}: {error_msg}")
            failed_pairs.append((pair_key, error_msg))
            print("⏭️  Continuing with next language pair...")
            continue

        # Print summary only if we have metrics
        if not has_metrics:
            continue

        print(f"\n📊 Summary for {pair_key}:")

        if args.mode is None:
            # Show comparison (both methods)
            if results.agent_avg_metrics and results.baseline_avg_metrics:
                # SER (lower is better)
                ser_agent = results.agent_avg_metrics.get("ser")
                ser_baseline = results.baseline_avg_metrics.get("ser")
                if ser_agent is not None and ser_baseline is not None:
                    print("   SER (Syllable Error Rate) ↓")
                    print(f"      Agent:      {ser_agent:.2%}")
                    print(f"      Baseline:   {ser_baseline:.2%}")
                    print(f"      Improvement: {ser_baseline - ser_agent:+.2%}")

                # SCRE (lower is better)
                scre_agent = results.agent_avg_metrics.get("scre")
                scre_baseline = results.baseline_avg_metrics.get("scre")
                if scre_agent is not None and scre_baseline is not None:
                    print("   SCRE (Syllable Count Relative Error) ↓")
                    print(f"      Agent:      {scre_agent:.2%}")
                    print(f"      Baseline:   {scre_baseline:.2%}")
                    print(f"      Improvement: {scre_baseline - scre_agent:+.2%}")

                # ARI (higher is better, range [-1, 1])
                ari_agent = results.agent_avg_metrics.get("ari")
                ari_baseline = results.baseline_avg_metrics.get("ari")
                if ari_agent is not None and ari_baseline is not None:
                    print("   ARI (Adjusted Rand Index) ↑")
                    print(f"      Agent:      {ari_agent:.3f}")
                    print(f"      Baseline:   {ari_baseline:.3f}")
                    print(f"      Improvement: {ari_agent - ari_baseline:+.3f}")
        elif args.mode == "agent":
            # Show agent only
            metrics = results.agent_avg_metrics
            if metrics:
                print("   SER (Syllable Error Rate) ↓")
                print(f"      {metrics.get('ser', 'N/A')}")
                print("   SCRE (Syllable Count Relative Error) ↓")
                print(f"      {metrics.get('scre', 'N/A')}")
                print("   ARI (Adjusted Rand Index) ↑")
                print(f"      {metrics.get('ari', 'N/A')}")
        else:  # args.mode == "base"
            # Show baseline only
            metrics = results.baseline_avg_metrics
            if metrics:
                print("   SER (Syllable Error Rate) ↓")
                print(f"      {metrics.get('ser', 'N/A')}")
                print("   SCRE (Syllable Count Relative Error) ↓")
                print(f"      {metrics.get('scre', 'N/A')}")
                print("   ARI (Adjusted Rand Index) ↑")
                print(f"      {metrics.get('ari', 'N/A')}")

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
    print(f"   Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
