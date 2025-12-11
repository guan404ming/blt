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
    DefaultPromptBuilder,
    MinimalPromptBuilder,
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
        model="gemini-2.5-pro",  # Use a different model
        auto_save=True,
        save_dir="custom_outputs",
        save_format="md",
        max_verification_rounds=20,
        enable_tool_logging=False,  # Disable verbose logging
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
    """Example 3: Using custom prompt builder"""
    print("\n" + "=" * 80)
    print("Example 3: Custom Prompt Builder")
    print("=" * 80)

    # Use minimal prompt builder for simpler prompts
    config = TranslatorConfig(model="gemini-2.5-flash")
    prompt_builder = MinimalPromptBuilder()

    translator = LyricsTranslator(config=config, prompt_builder=prompt_builder)

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

    from blt.translators import FeatureExtractor, ConstraintValidator

    # Create shared components
    feature_extractor = FeatureExtractor()
    validator = ConstraintValidator()

    # Create two translators sharing the same components
    translator1 = LyricsTranslator(
        config=TranslatorConfig(model="gemini-2.5-flash"),
        feature_extractor=feature_extractor,
        validator=validator,
    )

    translator2 = LyricsTranslator(
        config=TranslatorConfig(model="gemini-2.5-pro"),
        feature_extractor=feature_extractor,  # Shared!
        validator=validator,  # Shared!
    )

    lyrics = "Hello world"

    print("\nTranslator 1 (Flash):")
    result1 = translator1.translate(lyrics, "en-us", "cmn")
    print(f"  {result1.translated_lines[0]}")

    print("\nTranslator 2 (Pro):")
    result2 = translator2.translate(lyrics, "en-us", "cmn")
    print(f"  {result2.translated_lines[0]}")


if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)

    # Run examples
    example_basic_usage()
    example_custom_config()
    example_custom_prompt_builder()
    example_with_custom_constraints()
    example_reusing_components()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
