"""
Translate lyrics with music constraints
"""

import argparse
import logging
from pathlib import Path
from blt.translators import LyricsTranslationAgent, LyricsTranslationAgentConfig
from blt.translators import LyricsAnalyzer

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Load .env file from project root
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Translate lyrics with music constraints preservation"
    )
    parser.add_argument(
        "-f",
        "--lyrics-file",
        type=str,
        default="assets/lyrics-let-it-go.txt",
        help="Path to the lyrics file",
    )
    parser.add_argument(
        "-s",
        "--source-lang",
        type=str,
        default="en-us",
        help="Source language code (default: en-us)",
    )
    parser.add_argument(
        "-t",
        "--target-lang",
        type=str,
        default="cmn",
        help="Target language code (default: cmn)",
    )
    parser.add_argument(
        "-d",
        "--save-dir",
        type=str,
        default="outputs",
        help="Directory to save translation results",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model to use (default: qwen3:30b-a3b-instruct-2507-q4_K_M). Run 'ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M' first.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger("blt.translators").setLevel(logging.INFO)
    else:
        logging.getLogger("blt.translators").setLevel(logging.WARNING)

    # No API key needed for local Hugging Face models

    # Read lyrics
    lyrics_path = Path(args.lyrics_file)
    if not lyrics_path.exists():
        print(f"Error: Lyrics file not found: {args.lyrics_file}")
        return

    with open(lyrics_path, "r", encoding="utf-8") as f:
        source_lyrics = f.read().strip()

    print("=" * 80)
    print(f"Lyrics Translation: {lyrics_path.name}")
    print(f"Direction: {args.source_lang} → {args.target_lang}")
    print("=" * 80)
    print(f"\n【Source Lyrics】({args.source_lang})")
    print(source_lyrics)
    print()

    # Create config
    config = LyricsTranslationAgentConfig(
        model=args.model,
        auto_save=True,
        save_dir=args.save_dir,
        save_format="md",
        default_source_lang=args.source_lang,
        default_target_lang=args.target_lang,
        enable_logging=args.verbose,
    )

    # Create translator
    translator = LyricsTranslationAgent(config=config)

    # Extract target constraints
    analyzer = LyricsAnalyzer()
    constraints = analyzer.extract_constraints(source_lyrics, args.source_lang)

    # Translate
    print("\n" + "=" * 80)
    print("Translating...")
    print("=" * 80)

    result = translator.translate(source_lyrics)

    # Display results
    print("\n【Translation】")
    for i, line in enumerate(result.translated_lines, 1):
        print(f"{i}. {line}")

    print("\n【Syllables】")
    print(f"  Target: {constraints.syllable_counts}")
    print(f"  Actual: {result.syllable_counts}")
    print(f"  Match:  {result.syllable_counts == constraints.syllable_counts}")

    print("\n【Rhymes】")
    print(f"  Target scheme: {constraints.rhyme_scheme}")
    print(f"  Actual endings: {result.rhyme_endings}")

    if result.syllable_patterns and constraints.syllable_patterns:
        print("\n【Patterns】")
        for i, (target, actual) in enumerate(
            zip(constraints.syllable_patterns, result.syllable_patterns), 1
        ):
            # Calculate pattern similarity
            alignment = analyzer.analyze_pattern_alignment(target, actual)
            similarity = alignment["similarity"]
            match = "✓" if target == actual else "✗"

            # Display with score
            print(
                f"  {i}. Target: {target}  |  Actual: {actual}  {match}  ({similarity:.0%})"
            )

    if result.tool_call_stats:
        print("\n【Tool Calls】")
        for tool_name, count in sorted(result.tool_call_stats.items()):
            print(f"  • {tool_name}: {count}")

    print(f"\n【Reasoning】\n{result.reasoning}")


if __name__ == "__main__":
    main()
