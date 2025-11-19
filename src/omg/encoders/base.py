"""Base audio encoder interface."""

from abc import ABC, abstractmethod
import torch


class BaseAudioEncoder(ABC):
    """Abstract base class for audio encoders."""

    @abstractmethod
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: Audio waveform tensor [channels, samples] or [batch, channels, samples]
            sample_rate: Sample rate of the input audio

        Returns:
            Encoded tokens tensor
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to audio waveform.

        Args:
            tokens: Encoded tokens tensor

        Returns:
            Decoded audio waveform
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the encoder's expected sample rate.

        Returns:
            Sample rate in Hz
        """
        pass

    @abstractmethod
    def get_num_codebooks(self) -> int:
        """Get the number of codebooks used by the encoder.

        Returns:
            Number of codebooks
        """
        pass
