"""
Create soramimi (phonetic) translation of lyrics
"""

import os

# Configure phonemizer to use espeak-ng from ~/.local
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.expanduser("~/.local/bin/espeak-ng")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.expanduser(
    "~/.local/lib/libespeak-ng.so"
)

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from blt.translators import SoramimiTranslator, SoramimiConfig

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Create soramimi (phonetic) translation of lyrics"
    )
    parser.add_argument(
        "-f",
        "--lyrics-file",
        type=str,
        default="examples/lyrics-let-it-go.txt",
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
        help="Ollama model to use (default: qwen3:30b-a3b-instruct-2507-q4_K_M).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="IPA similarity threshold (default: 0.8)",
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
    print(f"Soramimi Translation: {lyrics_path.name}")
    print(f"Direction: {args.source_lang} -> {args.target_lang}")
    print(f"Similarity threshold: {args.threshold:.0%}")
    print("=" * 80)
    print(f"\n[Source Lyrics] ({args.source_lang})")
    print(source_lyrics)
    print()

    # Create config
    config = SoramimiConfig(
        model=args.model,
        auto_save=True,
        save_dir=args.save_dir,
        save_format="md",
        default_source_lang=args.source_lang,
        default_target_lang=args.target_lang,
        similarity_threshold=args.threshold,
        enable_logging=args.verbose,
    )

    # Create translator
    translator = SoramimiTranslator(config=config)

    # Translate
    print("\n" + "=" * 80)
    print("Creating soramimi translation...")
    print("=" * 80)

    result = translator.translate(source_lyrics)

    # Display results
    print("\n[Soramimi Translation]")
    for i, (line, score) in enumerate(
        zip(result.soramimi_lines, result.similarity_scores or []), 1
    ):
        status = "PASS" if (score and score >= args.threshold) else "FAIL"
        print(f"{i}. {line}  ({score:.1%} {status})" if score else f"{i}. {line}")

    if result.source_ipa and result.target_ipa and result.similarity_scores:
        print("\n[IPA Comparison]")
        for i, (src_ipa, tgt_ipa, score) in enumerate(
            zip(result.source_ipa, result.target_ipa, result.similarity_scores), 1
        ):
            print(f"Line {i}:")
            print(f"  Source: {src_ipa}")
            print(f"  Target: {tgt_ipa}")
            print(f"  Similarity: {score:.1%}")
            print()

    if result.overall_similarity:
        print(f"[Overall Similarity] {result.overall_similarity:.1%}")

    if result.tool_call_stats:
        print("\n[Tool Calls]")
        for tool_name, count in sorted(result.tool_call_stats.items()):
            print(f"  - {tool_name}: {count}")

    print(f"\n[Reasoning]\n{result.reasoning}")


if __name__ == "__main__":
    main()
