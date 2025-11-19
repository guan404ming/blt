"""MusicGen model implementation with ICL support."""

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .base import BaseMusicModel
from omg.encoders import BaseAudioEncoder, EnCodecEncoder


class MusicGenModel(BaseMusicModel):
    """MusicGen model wrapper for text-to-music generation with ICL support."""

    def __init__(
        self, model_name: str = "facebook/musicgen-small", device: str | None = None
    ):
        """Initialize the MusicGen model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self._sampling_rate = self.model.config.audio_encoder.sampling_rate

        # Initialize encoder for ICL
        self._encoder = EnCodecEncoder(
            model_name="facebook/encodec_32khz", device=self.device
        )

    def get_sampling_rate(self) -> int:
        """Get the model's audio sampling rate."""
        return self._sampling_rate

    def get_encoder(self) -> BaseAudioEncoder:
        """Get the audio encoder used by this model."""
        return self._encoder

    def _load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio tensor [1, samples]
        """
        audio, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self._sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self._sampling_rate)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio

    def generate(
        self,
        prompt: str,
        examples: list[tuple[str, str]] | None = None,
        duration: int = 8,
    ) -> torch.Tensor:
        """Generate music from a text prompt with optional audio examples.

        When examples are provided, the audio is encoded and used as context
        for the generation, enabling in-context learning.

        Args:
            prompt: Text description of the music to generate
            examples: Optional list of (description, audio_path) tuples for ICL
            duration: Approximate duration in seconds

        Returns:
            Generated audio tensor
        """
        if examples is not None and len(examples) > 0:
            # Encode all example audios and concatenate as context
            encoded_contexts = []
            example_descriptions = []

            for description, audio_path in examples:
                # Load audio
                audio = self._load_and_preprocess_audio(audio_path)

                # Encode to tokens using our encoder
                # Shape: [1, num_codebooks, seq_len]
                tokens = self._encoder.encode(audio, self._sampling_rate)
                encoded_contexts.append(tokens)
                example_descriptions.append(description)

            # Concatenate all encoded contexts along sequence dimension
            # Shape: [1, num_codebooks, total_seq_len]
            audio_context = torch.cat(encoded_contexts, dim=-1)

            # Decode back to waveform for use with generate
            # The model's generate_with_audio_conditioning expects raw audio
            context_audio = self._encoder.decode(audio_context)

            # Build combined prompt with example descriptions
            combined_prompt = " | ".join(example_descriptions) + " -> " + prompt

            # Process with audio conditioning
            inputs = self.processor(
                text=[combined_prompt],
                audio=context_audio.squeeze().cpu().numpy(),
                sampling_rate=self._sampling_rate,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Process text only
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate audio tokens (50 tokens ≈ 1 second)
        max_new_tokens = int(duration * 50)
        audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return audio_values[0, 0].cpu()
