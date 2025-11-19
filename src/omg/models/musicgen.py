"""MusicGen model implementation."""

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .base import BaseMusicModel


class MusicGenModel(BaseMusicModel):
    """MusicGen model wrapper for text-to-music generation."""

    def __init__(self, model_name: str = "facebook/musicgen-small"):
        """Initialize the MusicGen model.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self._sampling_rate = self.model.config.audio_encoder.sampling_rate

    def get_sampling_rate(self) -> int:
        """Get the model's audio sampling rate."""
        return self._sampling_rate

    def generate(
        self,
        prompt: str,
        example: tuple[str, str] | None = None,
        duration: int = 8,
    ) -> torch.Tensor:
        """Generate music from a text prompt with optional audio example.

        Args:
            prompt: Text description of the music to generate
            example: Optional tuple of (description, audio_path) for conditioning
            duration: Approximate duration in seconds

        Returns:
            Generated audio tensor
        """
        if example is not None:
            # Load and preprocess example audio
            _, audio_path = example
            audio, sr = torchaudio.load(audio_path)

            # Resample if necessary
            if sr != self._sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self._sampling_rate)
                audio = resampler(audio)

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Process with audio conditioning
            inputs = self.processor(
                text=[prompt],
                audio=audio.squeeze(0).numpy(),
                sampling_rate=self._sampling_rate,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Process text only
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

        # Generate audio tokens (50 tokens ≈ 1 second)
        max_new_tokens = int(duration * 50)
        audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return audio_values[0, 0]
