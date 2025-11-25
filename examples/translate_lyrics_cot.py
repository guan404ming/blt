"""
Translate Jay Chou's "擱淺" (Stranded) lyrics
Chinese → English translation with music constraints
"""

import os
from dotenv import load_dotenv
from omg.translators import LyricsTranslator, FeatureExtractor

# Load .env file
load_dotenv()


def main():
    # Load API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("請設定 GEMINI_API_KEY 環境變數")
        print("export GEMINI_API_KEY='your-api-key'")
        return

    # Read lyrics
    lyrics_file = "/home/gmchiu/Documents/Github/omg/擱淺.txt"
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

    extractor = FeatureExtractor(source_lang="Chinese", target_lang="English")
    constraints = extractor.extract_constraints(first_verse)

    print(f"\n音節數: {constraints.syllable_counts}")
    print(f"押韻方案: {constraints.rhyme_scheme}")
    print(f"停頓位置: {constraints.pause_positions}")

    # CoT translation
    print("\n" + "=" * 80)
    print("3. Chain-of-Thought 翻譯")
    print("=" * 80)

    translator_cot = LyricsTranslator(
        model="gemini-2.5-pro",
        api_key=api_key,
        use_cot=True,
        max_retries=1,
        auto_save=True,
        save_dir="outputs",
    )

    result_cot = translator_cot.translate(
        source_lyrics=first_verse,
        source_lang="Chinese",
        target_lang="English",
        save_format="json",  # Save as JSON
    )

    print("\n【Step 1: 意義分析】")
    print(result_cot.meaning_analysis)

    print("\n【Step 2: 約束分析】")
    for key, value in result_cot.constraint_analysis.items():
        print(f"  - {key}: {value}")

    print("\n【Step 3: 關鍵詞構思】")
    for keyword in result_cot.keyword_ideas:
        print(f"  - {keyword}")

    print("\n【Step 4: 最終翻譯】")
    for i, line in enumerate(result_cot.translated_lines, 1):
        print(f"{i}. {line}")

    print("\n【Step 5: 自我驗證】")
    for key, value in result_cot.self_verification.items():
        print(f"  - {key}: {value}")

    print("\n" + "=" * 80)
    print("翻譯完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
