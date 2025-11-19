"""Example script for MusicGen inference."""

from omg import generate_music


if __name__ == "__main__":
    # Text-only generation
    prompt = "happy electronic dance music with a catchy melody"
    generate_music(prompt)
