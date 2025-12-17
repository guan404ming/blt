"""Example usage of Gemini Translator

This example shows how to use the GeminiTranslator to translate lyrics
with constraints automatically extracted by the LyricsAnalyzer.
"""

import os
from blt.translators import GeminiTranslator, GeminiTranslatorConfig, LyricsAnalyzer

# Example: "See You Again" lyrics
EXAMPLE_LYRICS = """It's been a long day
without you, my friend

And I'll tell you all about it
when I see you again

We've come a long way
from where we began

Oh, I'll tell you all about it
when I see you again

When I see you again"""


def main():
    """Run example translation"""

    # Setup
    print("=" * 60)
    print("Gemini Lyrics Translator - Example")
    print("=" * 60)
    print()

    # Check API key (looks for GOOGLE_API_KEY or GEMINI_API_KEY)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: API key not found")
        print("   Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return

    # Initialize analyzer and translator
    analyzer = LyricsAnalyzer()
    config = GeminiTranslatorConfig(
        api_key=api_key,
        default_source_lang="en-us",
        default_target_lang="zh-tw",
        temperature=0.7,
        auto_save=False,  # Set to True to auto-save results
    )

    try:
        translator = GeminiTranslator(config)
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Install langchain-google-genai with: pip install langchain-google-genai")
        return

    print("📋 Source Lyrics (English):")
    print("-" * 60)
    print(EXAMPLE_LYRICS)
    print()

    # Extract constraints using analyzer
    print("🔍 Extracting constraints from source lyrics...")
    constraints = analyzer.extract_constraints(EXAMPLE_LYRICS, "en-us")

    print()
    print("🎯 Extracted Constraints:")
    print(f"  - Syllable counts: {constraints.syllable_counts}")
    print(f"  - Rhyme scheme: {constraints.rhyme_scheme}")
    if constraints.syllable_patterns:
        print(f"  - Syllable patterns: {len(constraints.syllable_patterns)} lines")
    print()

    # Translate
    print("🚀 Starting translation process...")
    print()

    result = translator.translate(
        source_lyrics=EXAMPLE_LYRICS,
        source_lang="en-us",
        target_lang="zh-tw",
        syllable_counts=constraints.syllable_counts,
        rhyme_scheme=constraints.rhyme_scheme,
        syllable_patterns=constraints.syllable_patterns,
    )

    # Display results
    print()
    print("=" * 60)
    print("✨ Translation Results")
    print("=" * 60)
    print()

    print("【Translation】")
    print()
    for i, line in enumerate(result.translated_lines, 1):
        print(f"{i}. {line}")

    print()
    print("【Syllables】")
    print(f"Target: {constraints.syllable_counts}")
    print(f"Actual: {result.syllable_counts}")

    print()
    print("【Rhymes】")
    print(f"Target scheme: {constraints.rhyme_scheme}")
    print(f"Actual: {result.rhyme_scheme}")

    print()
    print("【Validation】")
    for key, value in result.validation.items():
        print(f"  - {key}: {value}")

    if result.reasoning:
        print()
        print("【Reasoning】")
        print(result.reasoning[:500] + "..." if len(result.reasoning) > 500 else result.reasoning)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
