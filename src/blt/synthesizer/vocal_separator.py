"""Vocal separation module using Demucs for source separation."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Optional


class VocalSeparator:
    """Separates vocals from instrumental using Demucs model.

    Uses Meta's Demucs to separate audio into vocals and accompaniment.

    Args:
        model_name: Demucs model ('htdemucs', 'htdemucs_ft', 'mdx_extra')
        device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device

    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Separate vocals from instrumental.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated files

        Returns:
            Tuple of (vocals_path, instrumental_path)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory
        if output_dir is None:
            output_dir = audio_path.parent / "separated"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Separating audio using {self.model_name}...")

        # Create wrapper script to patch torchaudio before demucs runs
        wrapper = """
import torch
import torchaudio
import soundfile as sf

# Patch torchaudio.save to use soundfile instead of torchcodec
_orig_save = torchaudio.save

def patched_save(uri, src, sample_rate, **kwargs):
    audio = src.cpu().numpy() if isinstance(src, torch.Tensor) else src
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    sf.write(uri, audio, sample_rate)

torchaudio.save = patched_save

# Now run demucs
from demucs.separate import main
main()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            wrapper_path = f.name

        try:
            # Build command
            cmd = [
                sys.executable,
                wrapper_path,
                "-n",
                self.model_name,
                "-o",
                str(output_dir),
                "--two-stems",
                "vocals",
                str(audio_path),
            ]

            if self.device == "cpu":
                cmd.extend(["--device", "cpu"])

            # Run demucs with patched torchaudio
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Demucs failed: {e.stderr}")
        finally:
            Path(wrapper_path).unlink(missing_ok=True)

        # Find and move output files to output_dir root
        audio_name = audio_path.stem
        separated_dir = output_dir / self.model_name / audio_name

        src_vocals = separated_dir / "vocals.wav"
        src_instrumental = separated_dir / "no_vocals.wav"

        if not src_vocals.exists() or not src_instrumental.exists():
            raise FileNotFoundError(f"Output files not found in {separated_dir}")

        # Move files to output_dir
        vocals_path = output_dir / f"{audio_name}_vocals.wav"
        instrumental_path = output_dir / f"{audio_name}_instrumental.wav"

        shutil.move(str(src_vocals), str(vocals_path))
        shutil.move(str(src_instrumental), str(instrumental_path))

        # Clean up demucs directory structure
        shutil.rmtree(output_dir / self.model_name, ignore_errors=True)

        print("✓ Separation complete!")
        print(f"  Vocals: {vocals_path}")
        print(f"  Instrumental: {instrumental_path}")

        return str(vocals_path), str(instrumental_path)
