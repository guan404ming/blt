"""Base model interface for text-to-music generation."""

from abc import ABC, abstractmethod
import torch


class BaseMusicModel(ABC):
    """Abstract base class for music generation models."""

    @abstractmethod
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
        pass

    @abstractmethod
    def get_sampling_rate(self) -> int:
        """Get the model's audio sampling rate.

        Returns:
            Sampling rate in Hz
        """
        pass
