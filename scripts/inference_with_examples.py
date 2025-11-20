"""Example script for MusicGen inference with audio examples (ICL)."""

import json
from pathlib import Path

from omg import generate_music


if __name__ == "__main__":
    # Load examples from the examples directory
    examples_dir = Path(__file__).parent / "examples"

    # Load the example descriptions
    with open(examples_dir / "jazz.json") as f:
        example_data = json.load(f)

    # Build examples list as (description, audio_path) tuples
    examples = []
    for item in example_data:
        audio_path = examples_dir / item["file"]
        if audio_path.exists():
            examples.append((item["description"], str(audio_path)))
            print(f"Loaded example: {item['file']}")
        else:
            print(f"Warning: {audio_path} not found")

    # Generate with examples for in-context learning
    prompt = "jazz improvisation with saxophone and keyboard interplay"
    print(f"\nGenerating music with prompt: '{prompt}'")
    print(f"Using {len(examples)} audio examples for in-context learning\n")

    generate_music(
        prompt=prompt,
        examples=examples,
        output_path="scripts/examples/output_icl.wav",
        duration=20,  # Must be longer than total example audio duration
    )
