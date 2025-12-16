"""Example: Separate vocals and instrumental from audio using VocalSeparator.

This example demonstrates how to use the VocalSeparator class to split
an audio file into vocals and instrumental tracks.
"""

from blt.synthesizer import VocalSeparator


def main():
    # Input audio file
    input_audio = "examples/擱淺.mp3"

    # Output directory for separated files
    output_dir = "examples/separated_audio"

    print("=" * 60)
    print("VOCAL SEPARATION EXAMPLE")
    print("=" * 60)
    print(f"Input: {input_audio}")
    print(f"Output directory: {output_dir}")
    print()

    # Create separator
    separator = VocalSeparator(model_name="htdemucs")

    # Separate vocals and instrumental
    vocals_path, instrumental_path = separator.separate(
        audio_path=str(input_audio),
        output_dir=str(output_dir),
    )

    print()
    print("=" * 60)
    print("SEPARATION RESULTS")
    print("=" * 60)
    print(f"Vocals audio: {vocals_path}")
    print(f"Instrumental audio: {instrumental_path}")
    print()
    print("✓ Separation complete!")
    print()
    print("You can now use these files for:")
    print("  - Lyrics alignment with WhisperLyricsAligner")
    print("  - Voice synthesis or conversion")
    print("  - Creating karaoke tracks")


if __name__ == "__main__":
    main()
