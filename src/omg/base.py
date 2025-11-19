"""Model-agnostic music generation interface."""

import scipy.io.wavfile

from omg.models import BaseMusicModel, MusicGenModel


def generate_music(
    prompt: str,
    examples: list[tuple[str, str]] | None = None,
    output_path: str = "output.wav",
    duration: int = 8,
    model: BaseMusicModel | None = None,
):
    """Generate music from a text prompt with optional audio examples.

    Args:
        prompt: Text description of the music to generate
        examples: Optional list of (description, audio_path) tuples for in-context learning
        output_path: Path to save the generated audio
        duration: Approximate duration in seconds (50 tokens ≈ 1 second)
        model: Optional model instance. If None, uses MusicGenModel with default settings.

    Returns:
        Path to the generated audio file
    """
    # Use default model if none provided
    if model is None:
        model = MusicGenModel()

    # Generate audio
    audio_tensor = model.generate(prompt, examples=examples, duration=duration)

    # Save audio
    sampling_rate = model.get_sampling_rate()
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_tensor.numpy())

    print(f"Audio saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage without audio examples
    prompt = "happy electronic dance music with a catchy melody"
    generate_music(prompt)
