"""MusicGen model implementation with ICL support."""

import torch
import torchaudio
import soundfile as sf
from audiocraft.models import MusicGen

from .base import BaseMusicModel
from ..encoders.encodec import EnCodecEncoder


class MusicGenModel(BaseMusicModel):
    """MusicGen model wrapper for text-to-music generation with ICL support."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        encodec_model_name: str = "facebook/encodec_32khz",
        device: str | None = None,
    ):
        """Initialize the MusicGen model.

        Args:
            model_name: HuggingFace model name or path
            encodec_model_name: HuggingFace model name for EnCodec
            device: Device to run on. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MusicGen.get_pretrained(model_name, device=self.device)
        self.encoder = EnCodecEncoder(encodec_model_name, device=self.device)
        self._sampling_rate = self.model.sample_rate

    def get_sampling_rate(self) -> int:
        """Get the model's audio sampling rate."""
        return self._sampling_rate

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """Load audio file using soundfile to avoid FFmpeg dependency.

        Args:
            audio_path: Path to audio file

        Returns:
            (Audio tensor [channels, samples], sample_rate)
        """
        audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio_np).t()
        return audio, sr

    def generate(
        self,
        prompt: str,
        examples: list[tuple[str, str]] | None = None,
        duration: int = 8,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate music from a text prompt with optional audio examples.

        Args:
            prompt: Text description of the music to generate
            examples: Optional list of (description, audio_path) tuples for ICL
            duration: Approximate duration in seconds
            seed: Random seed for reproducibility

        Returns:
            Generated audio tensor
        """
        torch.manual_seed(seed)

        self.model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=duration,
        )

        if examples:
            example_prompts = []
            for description, audio_path in examples:
                audio, sr = self._load_audio(audio_path)
                # Encode audio to discrete tokens
                codes = self.encoder.encode(
                    audio, sr
                ).float()  # [1, num_codebooks, seq_len]

                # Create a compact, fixed-size representation using token statistics
                mean_tokens = codes.mean(dim=-1).squeeze()
                std_tokens = codes.std(dim=-1).squeeze()
                stats_vec = (
                    torch.cat([mean_tokens, std_tokens]).cpu().numpy().round(2).tolist()
                )
                stats_str = " ".join(map(str, stats_vec))

                example_prompts.append(
                    f'<example description="{description}" stats="{stats_str}">'
                )

            full_prompt = prompt + " " + " ".join(example_prompts)
        else:
            full_prompt = prompt

        # Use standard generate with the augmented prompt
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            audio_values = self.model.generate([full_prompt], progress=True)

        return audio_values[0].cpu().float().squeeze()
