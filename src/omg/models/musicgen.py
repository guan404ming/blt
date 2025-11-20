"""MusicGen model implementation with ICL support."""

import torch
import soundfile as sf
from audiocraft.models import MusicGen

from .base import BaseMusicModel
from ..encoders.encodec import EnCodecEncoder
from ..encoders.melody import MelodyEncoder


class MusicGenModel(BaseMusicModel):
    """MusicGen model wrapper for text-to-music generation with ICL support."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        encodec_model_name: str = "facebook/encodec_32khz",
        encoder_type: str = "encodec",
        device: str | None = None,
    ):
        """Initialize the MusicGen model.

        Args:
            model_name: HuggingFace model name or path
            encodec_model_name: HuggingFace model name for EnCodec
            encoder_type: Type of encoder to use ('encodec' or 'melody')
            device: Device to run on. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MusicGen.get_pretrained(model_name, device=self.device)
        self._sampling_rate = self.model.sample_rate

        if encoder_type == "encodec":
            self.prompt_encoder = EnCodecEncoder(encodec_model_name, device=self.device)
            self._embedding_tag = "stats"
        elif encoder_type == "melody":
            self.prompt_encoder = MelodyEncoder()
            self._embedding_tag = "melody"
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

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
                embedding_str = self.prompt_encoder.encode_to_string(audio, sr)

                example_prompts.append(
                    f'<example description="{description}" {self._embedding_tag}="{embedding_str}">'
                )

            full_prompt = prompt + "\n" + " ".join(example_prompts)
        else:
            full_prompt = prompt

        print(full_prompt)

        # Use standard generate with the augmented prompt
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            audio_values = self.model.generate([full_prompt], progress=True)

        return audio_values[0].cpu().float().squeeze()
