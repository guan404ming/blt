"""EnCodec audio encoder implementation using HuggingFace transformers."""

import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

from .base import BaseAudioEncoder


class EnCodecEncoder(BaseAudioEncoder):
    """EnCodec neural audio codec encoder.

    Uses the HuggingFace transformers implementation which is easy to load
    and compatible with MusicGen's internal encoder.
    """

    def __init__(self, model_name: str = "facebook/encodec_32khz", device: str | None = None):
        """Initialize the EnCodec encoder.

        Args:
            model_name: HuggingFace model name. Options:
                - "facebook/encodec_24khz" (24kHz, general audio)
                - "facebook/encodec_32khz" (32kHz, music - used by MusicGen)
                - "facebook/encodec_48khz" (48kHz, high quality)
            device: Device to run the model on. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Get sample rate from config
        self._sample_rate = self.model.config.sampling_rate
        self._num_codebooks = self.model.config.num_codebooks

    def get_sample_rate(self) -> int:
        """Get the encoder's expected sample rate."""
        return self._sample_rate

    def get_num_codebooks(self) -> int:
        """Get the number of codebooks used by the encoder."""
        return self._num_codebooks

    def _resample_if_needed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample audio to the encoder's sample rate if needed."""
        if sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self._sample_rate)
            audio = resampler(audio)
        return audio

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: Audio waveform tensor [channels, samples] or [batch, channels, samples]
            sample_rate: Sample rate of the input audio

        Returns:
            Encoded tokens tensor [batch, num_codebooks, seq_len]
        """
        # Handle different input shapes
        if audio.dim() == 2:
            # [channels, samples] -> [batch, channels, samples]
            audio = audio.unsqueeze(0)

        # Convert to mono if stereo
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        # Resample if needed
        audio = self._resample_if_needed(audio, sample_rate)

        # Move to device
        audio = audio.to(self.device)

        with torch.no_grad():
            # Encode using the model
            encoder_outputs = self.model.encode(audio, bandwidth=None)
            # encoder_outputs.audio_codes is [batch, num_codebooks, seq_len]
            codes = encoder_outputs.audio_codes

        return codes

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to audio waveform.

        Args:
            tokens: Encoded tokens tensor [batch, num_codebooks, seq_len]

        Returns:
            Decoded audio waveform [batch, channels, samples]
        """
        tokens = tokens.to(self.device)

        with torch.no_grad():
            # Decode using the model
            audio_values = self.model.decode(tokens, audio_scales=[None] * tokens.shape[0])

        return audio_values.audio_values
