"""Test script for lyrics translation using 擱淺.mp3 as example.

This script demonstrates the complete lyrics translation pipeline
using the provided audio file and lyrics.
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omg.lyrics_translation import LyricsTranslationPipeline


def main():
    # File paths
    audio_path = PROJECT_ROOT / "擱淺.mp3"
    old_lyrics_path = PROJECT_ROOT / "擱淺.txt"
    new_lyrics_path = PROJECT_ROOT / "擱淺_new.txt"

    # Check if files exist
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        print("Please ensure 擱淺.mp3 is in the project root directory.")
        sys.exit(1)

    if not old_lyrics_path.exists():
        print(f"Error: Old lyrics file not found: {old_lyrics_path}")
        sys.exit(1)

    if not new_lyrics_path.exists():
        print(f"Error: New lyrics file not found: {new_lyrics_path}")
        sys.exit(1)

    # Read lyrics
    with open(old_lyrics_path, "r", encoding="utf-8") as f:
        old_lyrics = f.read().strip()

    with open(new_lyrics_path, "r", encoding="utf-8") as f:
        new_lyrics = f.read().strip()

    print("=" * 60)
    print("LYRICS TRANSLATION TEST")
    print("=" * 60)
    print(f"\nAudio file: {audio_path.name}")
    print(f"\nOriginal lyrics:")
    print("-" * 60)
    print(old_lyrics)
    print("-" * 60)
    print(f"\nNew lyrics:")
    print("-" * 60)
    print(new_lyrics)
    print("-" * 60)

    # Initialize pipeline
    print("\n" + "=" * 60)
    print("INITIALIZING PIPELINE")
    print("=" * 60)

    pipeline = LyricsTranslationPipeline(
        separator_model="htdemucs",
        aligner_model="MahmoudAshraf/mms-300m-1130-forced-aligner",
        output_dir="lyrics_translation_output",
        device=None,  # Auto-detect
    )

    # Run pipeline
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE")
    print("=" * 60)

    try:
        results = pipeline.run(
            audio_path=str(audio_path),
            old_lyrics=old_lyrics,
            new_lyrics=new_lyrics,
            output_name="擱淺_cover",
        )

        # Print results
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files:")
        print(f"  Vocals:        {results['vocals']}")
        print(f"  Instrumental:  {results['instrumental']}")
        print(f"  Alignment:     {results['alignment']}")
        print(f"  New vocals:    {results['new_vocals']}")
        print(f"  Final cover:   {results['final_mix']}")
        print(f"  Metadata:      {results['metadata']}")
        print("\n" + "=" * 60)

        # Print alignment info
        if "word_timings" in results:
            word_timings = results["word_timings"]
            print(f"\nAligned {len(word_timings)} words from original lyrics")
            print("First 5 word timings:")
            for i, wt in enumerate(word_timings[:5]):
                print(
                    f"  {i + 1}. '{wt.word}' at {wt.start:.2f}s - {wt.end:.2f}s (score: {wt.score:.3f})"
                )

        print("\n" + "=" * 60)
        print("You can now listen to the generated cover song!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ ERROR OCCURRED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
