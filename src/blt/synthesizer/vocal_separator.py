"""Vocal separation module using Demucs for source separation."""

import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import subprocess
import tempfile
import sys

# Monkey-patch torchaudio.save to use soundfile backend instead of torchcodec
# This avoids the torchcodec dependency issue
import soundfile as sf

_original_torchaudio_save = torchaudio.save


def _soundfile_save(filepath, src, sample_rate, **kwargs):
    """Save audio using soundfile backend instead of torchcodec."""

    # Convert tensor to numpy and transpose if needed
    if isinstance(src, torch.Tensor):
        audio = src.cpu().numpy()
        # torchaudio uses (channels, samples), soundfile uses (samples, channels)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
    else:
        audio = src

    sf.write(filepath, audio, sample_rate)


# Apply the monkey patch
torchaudio.save = _soundfile_save


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

        # Create a wrapper script that patches torchaudio.save before running demucs
        wrapper_script = '''
import sys
import torch
import torchaudio
import soundfile as sf

# Monkey-patch torchaudio.save to use soundfile instead of torchcodec
def _soundfile_save(filepath, src, sample_rate, **kwargs):
    """Save audio using soundfile backend instead of torchcodec."""
    # Convert tensor to numpy
    if isinstance(src, torch.Tensor):
        audio = src.cpu().numpy()
        # torchaudio uses (channels, samples), soundfile uses (samples, channels)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
    else:
        audio = src
    sf.write(filepath, audio, sample_rate)

torchaudio.save = _soundfile_save

# Now run demucs
from demucs.separate import main
main()
'''

        # Write the wrapper script to a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_script)
            wrapper_path = f.name

        try:
            # Build command to run demucs through our wrapper
            cmd = [
                sys.executable,
                wrapper_path,
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
        finally:
            # Clean up the wrapper script
            import os

            try:
                os.unlink(wrapper_path)
            except:
                pass

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

        print("Separation complete!")
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
