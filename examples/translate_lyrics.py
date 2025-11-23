"""
Example: Translate lyrics using PydanticAI + Gemini 2.0 Flash
範例: 使用 PydanticAI + Gemini 2.0 Flash 翻譯歌詞
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lyrics_translation import LyricsTranslator, FeatureExtractor


def main():
    # 設定 API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("請設定 GEMINI_API_KEY 環境變數")
        print("export GEMINI_API_KEY='your-api-key'")
        return

    # 範例歌詞 (Let It Go 片段)
    source_lyrics = """Let it go, let it go
Can't hold it back anymore
Let it go, let it go
Turn away and slam the door"""

    print("=" * 60)
    print("歌詞翻譯範例: Let It Go (英文 → 中文)")
    print("=" * 60)
    print("\n【原始歌詞】")
    print(source_lyrics)
    print()

    # 1. Zero-shot 翻譯 (基礎版)
    print("\n" + "="*60)
    print("1. Zero-shot 翻譯 (無 CoT)")
    print("="*60)

    translator = LyricsTranslator(
        model="gemini-2.5-flash",
        api_key=api_key,
        use_cot=False,
        max_retries=3,
        auto_save=True,
        save_dir="outputs"
    )

    result = translator.translate(
        source_lyrics=source_lyrics,
        source_lang="English",
        target_lang="Chinese",
        save_format="txt"  # Save as plain text
    )

    print("\n【翻譯結果】")
    for i, line in enumerate(result.translated_lines, 1):
        print(f"{i}. {line}")

    print(f"\n【音節數】{result.syllable_counts}")
    print(f"【韻腳】{result.rhyme_endings}")
    print(f"\n【翻譯思路】\n{result.reasoning}")

    # 2. CoT 翻譯 (進階版)
    print("\n" + "="*60)
    print("2. Chain-of-Thought 翻譯")
    print("="*60)

    translator_cot = LyricsTranslator(
        model="gemini-2.5-flash",
        api_key=api_key,
        use_cot=True,
        max_retries=3,
        auto_save=True,
        save_dir="outputs"
    )

    result_cot = translator_cot.translate(
        source_lyrics=source_lyrics,
        source_lang="English",
        target_lang="Chinese",
        save_format="md"  # Save as Markdown
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

    # 3. 顯示自動提取的約束
    print("\n" + "="*60)
    print("3. 自動提取的音樂約束")
    print("="*60)

    extractor = FeatureExtractor(source_lang="English", target_lang="Chinese")
    constraints = extractor.extract_constraints(source_lyrics)

    print(f"\n音節數: {constraints.syllable_counts}")
    print(f"押韻方案: {constraints.rhyme_scheme}")
    print(f"停頓位置: {constraints.pause_positions}")


if __name__ == "__main__":
    main()
