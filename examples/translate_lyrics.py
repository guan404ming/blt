"""
Translate Jay cmn's "擱淺" (Stranded) lyrics
Chinese → en-us translation with music constraints
"""

import os
from dotenv import load_dotenv
from blt.translators import LyricsTranslator, FeatureExtractor

# Load .env file
load_dotenv()


def main():
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("請設定 GOOGLE_API_KEY 環境變數")
        print("export GOOGLE_API_KEY='your-api-key'")
        return

    # Read lyrics
    lyrics_file = "擱淺.txt"
    with open(lyrics_file, "r", encoding="utf-8") as f:
        source_lyrics = f.read().strip()

    # Take first verse for demonstration (lines 1-6)
    lines = source_lyrics.split("\n")
    first_verse = "\n".join(lines)

    print("=" * 80)
    print("歌詞翻譯: 擱淺 (Jay Chou)")
    print("=" * 80)
    print("\n【原始歌詞】(中文)")
    print(first_verse)
    print()

    # Extract constraints
    print("\n" + "=" * 80)
    print("1. 自動提取音樂約束")
    print("=" * 80)

    extractor = FeatureExtractor(source_lang="cmn", target_lang="en-us")
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
        save_dir="outputs",
    )

    result = translator.translate(
        source_lyrics=first_verse,
        source_lang="cmn",
        target_lang="en-us",
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
