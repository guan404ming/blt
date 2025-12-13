"""
Example: Using the Modular Lyrics Translator

This example demonstrates the new modular architecture with:
1. Configuration objects
2. Custom prompt builders
3. Dependency injection
4. Reusable components
"""

import os
from dotenv import load_dotenv
from blt.translators import (
    LyricsTranslator,
    TranslatorConfig,
)

load_dotenv()


def example_basic_usage():
    """Example 1: Basic usage with defaults"""
    print("=" * 80)
    print("Example 1: Basic Usage with Defaults")
    print("=" * 80)

    # Simple initialization - uses all defaults
    translator = LyricsTranslator()

    lyrics = """I love you
You love me"""

    result = translator.translate(
        source_lyrics=lyrics,
        source_lang="en-us",
        target_lang="cmn",
    )

    print("\nTranslation:")
    for line in result.translated_lines:
        print(f"  {line}")


def example_custom_config():
    """Example 2: Using custom configuration"""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Create custom configuration
    config = TranslatorConfig(
        model="qwen3:30b-a3b-instruct-2507-q4_K_M",  # Use Ollama model
        auto_save=True,
        save_dir="custom_outputs",
        save_format="md",
        max_retries=20,
        enable_logging=False,  # Disable verbose logging
    )

    translator = LyricsTranslator(config=config)

    lyrics = """Hello world
Nice to meet you"""

    result = translator.translate(
        source_lyrics=lyrics,
        source_lang="en-us",
        target_lang="cmn",
    )

    print("\nTranslation:")
    for line in result.translated_lines:
        print(f"  {line}")


def example_custom_prompt_builder():
    """Example 3: Using different models"""
    print("\n" + "=" * 80)
    print("Example 3: Using Different Models")
    print("=" * 80)

    # Use a faster model configuration
    config = TranslatorConfig(
        model="qwen3:30b-a3b-instruct-2507-q4_K_M",
        max_retries=10,
    )

    translator = LyricsTranslator(config=config)

    lyrics = """Good morning
Have a nice day"""

    result = translator.translate(
        source_lyrics=lyrics,
        source_lang="en-us",
        target_lang="cmn",
    )

    print("\nTranslation:")
    for line in result.translated_lines:
        print(f"  {line}")


def example_with_custom_constraints():
    """Example 4: Providing custom constraints"""
    print("\n" + "=" * 80)
    print("Example 4: Custom Constraints")
    print("=" * 80)

    from blt.translators import MusicConstraints

    # Create custom constraints
    constraints = MusicConstraints(
        syllable_counts=[3, 4],
        rhyme_scheme="AA",
        syllable_patterns=[[1, 2], [1, 1, 2]],
    )

    translator = LyricsTranslator()

    lyrics = """I love you
You love me too"""

    result = translator.translate(
        source_lyrics=lyrics,
        source_lang="en-us",
        target_lang="cmn",
        constraints=constraints,  # Use custom constraints
    )

    print("\nTranslation:")
    for line in result.translated_lines:
        print(f"  {line}")

    print(f"\nSyllable counts: {result.syllable_counts}")
    print(f"Rhyme endings: {result.rhyme_endings}")


def example_reusing_components():
    """Example 5: Reusing components across multiple translators"""
    print("\n" + "=" * 80)
    print("Example 5: Reusing Components")
    print("=" * 80)

    from blt.translators import LyricsAnalyzer

    # Create shared analyzer component
    analyzer = LyricsAnalyzer()

    # Create two translators sharing the same analyzer
    translator1 = LyricsTranslator(
        config=TranslatorConfig(model="qwen3:30b-a3b-instruct-2507-q4_K_M"),
        analyzer=analyzer,
    )

    translator2 = LyricsTranslator(
        config=TranslatorConfig(model="qwen3:30b-a3b-instruct-2507-q4_K_M"),
        analyzer=analyzer,  # Shared analyzer!
    )

    lyrics = "Hello world"

    print("\nTranslator 1 (Flash):")
    result1 = translator1.translate(lyrics, "en-us", "cmn")
    print(f"  {result1.translated_lines[0]}")

    print("\nTranslator 2 (Pro):")
    result2 = translator2.translate(lyrics, "en-us", "cmn")
    print(f"  {result2.translated_lines[0]}")


if __name__ == "__main__":
    # No API key needed for local Hugging Face models

    # Run examples
    example_basic_usage()
    example_custom_config()
    example_custom_prompt_builder()
    example_with_custom_constraints()
    example_reusing_components()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
