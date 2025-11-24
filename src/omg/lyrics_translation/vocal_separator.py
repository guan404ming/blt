"""Vocal separation module using Demucs for source separation."""

import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import subprocess
import tempfile
import sys


class VocalSeparator:
    """Separates vocals from instrumental using Demucs model.

    This class uses Meta's Demucs (Deep Extractor for Music Sources) model
    to separate audio into vocals and accompaniment (instrumental).

    Args:
        model_name: Demucs model to use. Options: 'htdemucs', 'htdemucs_ft', 'mdx_extra'
        device: Device to run the model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Separate vocals from instrumental.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated files. If None, uses temp directory.

        Returns:
            Tuple of (vocals_path, instrumental_path)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="vocal_separation_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Run Demucs separation
        print(f"Separating audio using {self.model_name} model...")
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            self.model_name,
            "-o",
            str(output_dir),
            "--two-stems",
            "vocals",  # Only separate vocals and instrumental
            str(audio_path),
        ]

        if self.device == "cpu":
            cmd.extend(["--device", "cpu"])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Demucs separation failed: {e.stderr}") from e

        # Find output files
        # Demucs creates: output_dir / model_name / audio_name / vocals.wav
        audio_name = audio_path.stem
        separated_dir = output_dir / self.model_name / audio_name

        vocals_path = separated_dir / "vocals.wav"
        instrumental_path = separated_dir / "no_vocals.wav"

        if not vocals_path.exists():
            raise FileNotFoundError(
                f"Demucs did not create vocals file at {vocals_path}"
            )
        if not instrumental_path.exists():
            raise FileNotFoundError(
                f"Demucs did not create instrumental file at {instrumental_path}"
            )

        print(f"Separation complete!")
        print(f"  Vocals: {vocals_path}")
        print(f"  Instrumental: {instrumental_path}")

        return str(vocals_path), str(instrumental_path)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file as tensor.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate

    def save_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        output_path: str,
    ):
        """Save audio tensor to file.

        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of audio
            output_path: Path to save audio file
        """
        torchaudio.save(output_path, waveform, sample_rate)
        print(f"Saved audio to {output_path}")
