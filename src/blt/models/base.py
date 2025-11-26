"""Base model interface for text-to-music generation."""

from abc import ABC, abstractmethod
import torch


class BaseMusicModel(ABC):
    """Abstract base class for music generation models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        examples: list[tuple[str, str]] | None = None,
        duration: int = 8,
    ) -> torch.Tensor:
        """Generate music from a text prompt with optional example descriptions.

        Args:
            prompt: Text description of the music to generate
            examples: Optional list of (description, audio_path) tuples.
                      Only descriptions are used to build the prompt.
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
