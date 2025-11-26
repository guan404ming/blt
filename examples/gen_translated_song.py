"""Demo script for lyrics translation pipeline.

This script demonstrates how to use the lyrics translation pipeline
to generate a cover song with new lyrics while maintaining the original
singer's voice and instrumental.

Usage:
    uv run python scripts/lyrics_translation_demo.py \\
        --audio path/to/song.mp3 \\
        --old-lyrics "Original lyrics here" \\
        --new-lyrics "New lyrics here"
"""

import argparse
from pathlib import Path
import sys
import os
from omg.pipeline import CoverSongPipeline

# Disable torchcodec backend for torchaudio to avoid dependency
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# Prevent laion_clap from parsing command line arguments during import
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]


def main():
    # Restore the original sys.argv for argument parsing
    global _original_argv
    sys.argv = _original_argv

    parser = argparse.ArgumentParser(description="Lyrics Translation Pipeline Demo")

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to original song audio file",
    )

    parser.add_argument(
        "--old-lyrics",
        type=str,
        help="Original lyrics text (inline)",
    )

    parser.add_argument(
        "--old-lyrics-file",
        type=str,
        help="Path to file containing original lyrics",
    )

    parser.add_argument(
        "--new-lyrics",
        type=str,
        help="New lyrics text (inline)",
    )

    parser.add_argument(
        "--new-lyrics-file",
        type=str,
        help="Path to file containing new lyrics",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="lyrics_translation_output",
        help="Directory to save outputs (default: lyrics_translation_output)",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        help="Name for output files (default: uses input audio filename)",
    )

    parser.add_argument(
        "--separator-model",
        type=str,
        default="htdemucs",
        help="Demucs model for vocal separation (default: htdemucs)",
    )

    parser.add_argument(
        "--aligner-model",
        type=str,
        default="MahmoudAshraf/mms-300m-1130-forced-aligner",
        help="Model for lyrics alignment (default: MahmoudAshraf/mms-300m-1130-forced-aligner)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run models on (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate inputs
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Load lyrics
    if args.old_lyrics:
        old_lyrics = args.old_lyrics
    elif args.old_lyrics_file:
        with open(args.old_lyrics_file, "r", encoding="utf-8") as f:
            old_lyrics = f.read().strip()
    else:
        print("Error: Must provide either --old-lyrics or --old-lyrics-file")
        sys.exit(1)

    if args.new_lyrics:
        new_lyrics = args.new_lyrics
    elif args.new_lyrics_file:
        with open(args.new_lyrics_file, "r", encoding="utf-8") as f:
            new_lyrics = f.read().strip()
    else:
        print("Error: Must provide either --new-lyrics or --new-lyrics-file")
        sys.exit(1)

    # Initialize pipeline
    print("\n" + "=" * 60)
    print("COVER SONG GENERATION DEMO")
    print("=" * 60)

    pipeline = CoverSongPipeline(
        separator_model=args.separator_model,
        aligner_model=args.aligner_model,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run pipeline
    results = pipeline.run(
        audio_path=str(audio_path),
        old_lyrics=old_lyrics,
        new_lyrics=new_lyrics,
        output_name=args.output_name,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Separated vocals: {results['vocals']}")
    print(f"Separated instrumental: {results['instrumental']}")
    print(f"Lyrics alignment: {results['alignment']}")
    print(f"New vocals: {results['new_vocals']}")
    print(f"Final cover: {results['final_mix']}")
    print(f"Metadata: {results['metadata']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
