"""Example script for MusicGen inference."""

from omg import generate_music


if __name__ == "__main__":
    # Text-only generation
    prompt = "jazz improvisation with saxophone and keyboard interplay"
    generate_music(
        prompt,
        duration=20,
        output_path="scripts/examples/output_baseline.wav",
    )
