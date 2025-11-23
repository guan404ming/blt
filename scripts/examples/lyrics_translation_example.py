"""Simple example of lyrics translation pipeline.

This example shows how to use the lyrics translation pipeline
with minimal code.
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omg.lyrics_translation import LyricsTranslationPipeline


def example_basic():
    """Basic example with inline lyrics."""

    # Example lyrics (simplified for demo)
    old_lyrics = """
    Twinkle twinkle little star
    How I wonder what you are
    Up above the world so high
    Like a diamond in the sky
    """

    new_lyrics = """
    Sparkle sparkle tiny light
    Shining brightly through the night
    Dancing in the sky so free
    Bringing magic down to me
    """

    # Initialize pipeline
    pipeline = LyricsTranslationPipeline(
        output_dir="example_output"
    )

    # Note: You need to provide your own audio file
    audio_path = "path/to/your/song.wav"

    print("⚠️  Please provide an audio file path in the script")
    print("Example usage:")
    print(f"  audio_path = '{audio_path}'")
    print("\nThen run the pipeline:")
    print("  results = pipeline.run(")
    print("      audio_path=audio_path,")
    print("      old_lyrics=old_lyrics,")
    print("      new_lyrics=new_lyrics,")
    print("  )")


def example_from_files():
    """Example loading lyrics from files."""

    pipeline = LyricsTranslationPipeline(
        output_dir="example_output"
    )

    # Load lyrics from files
    with open("old_lyrics.txt", "r", encoding="utf-8") as f:
        old_lyrics = f.read()

    with open("new_lyrics.txt", "r", encoding="utf-8") as f:
        new_lyrics = f.read()

    # Run pipeline
    results = pipeline.run(
        audio_path="path/to/song.wav",
        old_lyrics=old_lyrics,
        new_lyrics=new_lyrics,
        output_name="my_cover",
    )

    print(f"Cover song generated: {results['final_mix']}")


if __name__ == "__main__":
    print("=" * 60)
    print("LYRICS TRANSLATION EXAMPLE")
    print("=" * 60)
    print("\nThis example demonstrates the lyrics translation pipeline.")
    print("Please edit this file to provide your audio file path.")
    print("\nFor a complete demo, use:")
    print("  python scripts/lyrics_translation_demo.py --help")
    print("=" * 60)

    example_basic()
