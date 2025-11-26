"""
Translate lyrics with music constraints
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from blt.translators import LyricsTranslator, FeatureExtractor

# Load .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Translate lyrics with music constraints preservation"
    )
    parser.add_argument(
        "-f", "--lyrics-file",
        type=str,
        default="/Users/wchiu/Documents/GitHub/blt/examples/lyrics-let-it-go.txt",
        help="Path to the lyrics file (default: lyrics-let-it-go.txt)",
    )
    parser.add_argument(
        "-s", "--source-lang",
        type=str,
        default="eng",
        help="Source language code (default: eng)",
    )
    parser.add_argument(
        "-t", "--target-lang",
        type=str,
        default="cmn",
        help="Target language code (default: cmn)",
    )
    parser.add_argument(
        "-d", "--save-dir",
        type=str,
        default="outputs",
        help="Directory to save translation results (default: outputs)",
    )

    args = parser.parse_args()

    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("請設定 GOOGLE_API_KEY 環境變數")
        print("export GOOGLE_API_KEY='your-api-key'")
        return

    # Read lyrics
    lyrics_path = Path(args.lyrics_file)
    if not lyrics_path.exists():
        print(f"Error: Lyrics file not found: {args.lyrics_file}")
        return

    with open(lyrics_path, "r", encoding="utf-8") as f:
        source_lyrics = f.read().strip()

    # Take first verse for demonstration
    lines = source_lyrics.split("\n")
    first_verse = "\n".join(lines)

    print("=" * 80)
    print(f"歌詞翻譯: {lyrics_path.name}")
    print(f"翻譯方向: {args.source_lang} → {args.target_lang}")
    print("=" * 80)
    print(f"\n【原始歌詞】({args.source_lang})")
    print(first_verse)
    print()

    # Extract constraints
    print("\n" + "=" * 80)
    print("1. 自動提取音樂約束")
    print("=" * 80)

    extractor = FeatureExtractor(source_lang=args.source_lang, target_lang=args.target_lang)
    constraints = extractor.extract_constraints(first_verse)

    print(f"\n音節數: {constraints.syllable_counts}")
    print(f"押韻方案: {constraints.rhyme_scheme}")
    print(f"停頓位置: {constraints.pause_positions}")

    # Zero-shot translation
    print("\n" + "=" * 80)
    print("2. Zero-shot 翻譯")
    print("=" * 80)

    translator = LyricsTranslator(
        api_key=api_key,
        auto_save=True,
        save_dir=args.save_dir,
    )

    result = translator.translate(
        source_lyrics=first_verse,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        save_format="md",  # Save as Markdown
    )

    print("\n【翻譯結果】")
    for i, line in enumerate(result.translated_lines, 1):
        print(f"{i}. {line}")

    print(f"\n【音節數】{result.syllable_counts}")
    print(f"【韻腳】{result.rhyme_endings}")
    print(f"\n【翻譯思路】\n{result.reasoning}")


if __name__ == "__main__":
    main()
